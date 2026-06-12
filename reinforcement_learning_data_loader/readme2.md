# Unified Post-Training Data Loader — File Guide

This document explains, file by file, what each component of the data
loader does and how it fits into the overall pipeline.

---

## `dataset.h` — Data layer (mode-aware)

This is where raw bytes become tensors, and the only place (besides
`collator.h`) that knows about training modes.

- **`TrainingMode` enum** — `SFT`, `REWARD_MODEL`, `DPO`, `ROLLOUT`. This
  single flag drives everything downstream.

- **`RawSample` struct** — a unified container with four optional fields:
  `prompt_ids` (always populated), `response_ids` (SFT only),
  `chosen_ids`/`rejected_ids` (REWARD_MODEL/DPO only). One struct type
  flows through the whole pipeline regardless of mode; unused fields are
  just empty vectors.

- **`DataLoaderConfig`** — the single source of truth. Universal knobs
  (`batch_size`, `num_workers`, `shuffle`, `seed`, bucket boundaries,
  pad/eos/bos token ids), mode-specific knobs (`max_seq_len` for
  SFT/RM/DPO, `max_prompt_len`/`max_gen_len` for ROLLOUT), the
  `grpo_group_size` field (stored for the training loop's benefit only —
  never read by the loader itself), and the four file paths.

- **`BinarySequenceReader`** — the lowest-level reader. Reads the format
  `uint32 n_samples` then per-sample `uint16 length` + `uint32
  token_ids[length]`. Returns `vector<vector<int32_t>>`.

- **`UnifiedDatasetReader::load(cfg)`** — dispatches on `cfg.mode`:
  - `SFT` → reads `prompts.bin` + `responses.bin`, zips into `RawSample`s
  - `REWARD_MODEL`/`DPO` → reads `prompts.bin` + `chosen.bin` +
    `rejected.bin` (same loading code for both modes — they only diverge
    later, in the collator)
  - `ROLLOUT` → reads only `prompts.bin`

  This is the "smart" entry point: it never opens a file the current mode
  doesn't need.

- **`UnifiedDataset`** — wraps the loaded samples and does two things:

  1. Pre-computes `lengths_[]` at construction time, with a different
     definition per mode (this feeds the bucket sampler later):
     - SFT: `min(prompt+response, max_seq_len)`
     - REWARD_MODEL/DPO: roughly `prompt_len + max(chosen_len,
       rejected_len)`, capped
     - ROLLOUT: `min(prompt_len, max_prompt_len)`

  2. Exposes three `get_*` methods, one per item shape:
     - **`get_sft(idx)`** → `SFTItem{prompt_ids, response_ids}`.
       Truncation logic (`truncate_sft`) protects the response: it gets
       up to half the budget first, then the prompt gets squeezed into
       whatever's left, restoring the EOS token at the cut point.
     - **`get_preference(idx)`** → `PreferenceItem{prompt_ids,
       chosen_ids, rejected_ids}`. Chosen and rejected are truncated
       independently via `truncate_one_side`, each fit into `max_seq_len
       - prompt_len`. The prompt itself is also clamped to a quarter of
       the budget if it's huge.
     - **`get_rollout(idx)`** → `RolloutItem{prompt_ids, prompt_mask}`.
       The prompt is right-truncated to `max_prompt_len`, then the whole
       thing is built as a fixed-size `(max_prompt_len + max_gen_len)`
       vector with left padding: padding tokens first, then the real
       prompt tokens, then zeros reserved for generation. This is done
       per-item here (not in the collator), so the collator just stacks.

- **Legacy compatibility** — `BinaryDatasetReader::load(instructions_path,
  responses_path)` and `InstructionDataset` are thin wrappers that
  internally build a `DataLoaderConfig` with `mode = SFT` and delegate to
  `UnifiedDatasetReader`/`UnifiedDataset`. This means old SFT-only code
  that calls these names still compiles and works.

---

## `collator.h` — Mode-specific batching

Takes a list of items (from `dataset.h`) and turns them into padded
GPU-ready tensors. This is the second and last mode-aware file.

**Batch structs:**

- `SFTBatch{input_ids, attention_mask, labels}` — `labels` has `-100`
  over the prompt portion, real token ids over the response. Aliased as
  `Batch` for backward compatibility.
- `RewardModelBatch{chosen_input_ids, chosen_attention_mask,
  rejected_input_ids, rejected_attention_mask}` — no labels tensor at
  all, since the reward model only reads the scalar at the last token
  position.
- `DPOBatch{chosen_input_ids, chosen_attention_mask, chosen_labels,
  rejected_input_ids, rejected_attention_mask, rejected_labels}` — same
  shape as `RewardModelBatch` plus per-side labels, because DPO needs
  token-level log-probs.
- `RolloutBatch{prompt_ids, prompt_mask}` — shape `(B, max_prompt_len +
  max_gen_len)`.

**`DynamicPaddingCollator`** — the original SFT-only collator, kept
verbatim for any code still calling it directly with `(instruction,
response)` tensor pairs.

**`detail::pad_sequences`** — helper that takes a vector of
variable-length 1D tensors and right-pads them to a single 2D `(ids,
mask)` pair. Used by both reward-model and DPO collation for the chosen
side and rejected side independently (so `L_chosen` and `L_rejected` can
differ).

**`UnifiedCollator`** — constructed once with the config, exposes four
methods:

- **`collate_sft(items)`** — concatenates `prompt + response` per sample,
  builds labels by prepending `-100` over the prompt length, pads
  everything to the batch max length.
- **`collate_reward_model(items)`** — for each sample builds
  `prompt+chosen` and `prompt+rejected` as two separate sequences, pads
  each side independently via `pad_sequences`, returns `RewardModelBatch`
  with no labels.
- **`collate_dpo(items)`** — does the same concatenation/padding as
  reward-model, then additionally builds `chosen_labels`/`rejected_labels`
  by copying the real token ids past the prompt length and leaving `-100`
  everywhere else (including padding).
- **`collate_rollout(items)`** — items arrive already left-padded and
  fixed-size from `get_rollout`, so this just `unsqueeze(0)` + `cat` them
  into a batch.

---

## `sampler.h` — Mode-agnostic bucketing

Has zero knowledge of training modes — it only ever sees a `lengths`
array and index counts.

- **`shard_indices(total, rank, world_size, shuffle, seed, epoch)`** —
  for multi-GPU training, builds a shuffled (seeded by `seed+epoch`, so
  each epoch differs) index list, pads it to a multiple of `world_size`,
  and returns the slice belonging to this rank. Each GPU gets a disjoint
  subset.

- **`BucketBatchSampler`** — given the dataset's `lengths()` (whatever
  that means for the active mode), the shard of indices, batch size, and
  bucket boundaries:
  1. Assigns each index to a length bucket (e.g. ≤64, ≤128, ≤256, ...).
  2. Within each bucket, optionally shuffles, then chops into
     `batch_size`-sized groups.
  3. Optionally shuffles the resulting batch order.

  Result: `batches()` returns `vector<vector<int64_t>>` — a list of
  batches, each a list of sample indices, where same-batch samples have
  similar lengths (minimizing padding waste). Because the meaning of
  "length" changes per mode but the sampler only consumes numbers, no
  mode-specific code is needed here.

---

## `prefetcher.h` — Background GPU delivery

**`detail::to_device` overloads** — one per batch type (`SFTBatch`,
`RewardModelBatch`, `DPOBatch`, `RolloutBatch`), each pinning memory and
async-copying every tensor field to the target device. The base overload
handles a single `torch::Tensor`.

**`TypedPrefetcher<BatchT>`** — a generic background-worker pool:

- Constructor takes a `CollatorFn` (a callable `vector<int64_t> indices ->
  BatchT`, built by `dataloader.h`), a `BucketBatchSampler`, a device,
  worker count, and prefetch factor.
- Spawns `num_workers` threads. Each worker pulls the next batch's
  indices, calls `collate_fn_` (which internally calls `dataset.get_*` +
  `collator.collate_*`), moves the result to GPU via `to_device`, and
  pushes it onto a shared queue — staying `prefetch_factor` batches ahead.
- `.next()` blocks only if the queue is empty and the epoch isn't done
  yet; returns `optional<BatchT>`, `nullopt` at epoch end.

**Type aliases** — `SFTPrefetcher`, `RewardModelPrefetcher`,
`DPOPrefetcher`, `RolloutPrefetcher` are `TypedPrefetcher<XBatch>`
instantiations. `CUDAPrefetcher = SFTPrefetcher` is the legacy alias.

This file is "lightly mode-aware" only in the sense that it's templated
per batch type — its threading/queueing logic itself never branches on
mode.

---

## `dataloader.h` — The factory / wiring layer

This is the "smart entry point" that ties everything above together.

- **`detail::make_sampler(dataset, cfg, rank, world_size, epoch)`** —
  calls `shard_indices` then constructs a `BucketBatchSampler` from
  `dataset.lengths()`. Shared by all four factories.

- **`build_sft_dataloader` / `build_reward_model_dataloader` /
  `build_dpo_dataloader` / `build_rollout_dataloader`** — each one:
  1. Builds the sampler via the helper above.
  2. Constructs a `UnifiedCollator`.
  3. Builds a `CollatorFn` lambda that, for a given list of indices, calls
     the right `dataset->get_*(idx)` for each index and feeds the
     resulting items to the right `collator->collate_*(items)`.
  4. Wraps it all in the matching `TypedPrefetcher<BatchT>` and returns a
     `unique_ptr` to it.

  Each factory returns a different concrete type (e.g.
  `unique_ptr<SFTPrefetcher>` vs `unique_ptr<RolloutPrefetcher>`) —
  deliberately, since the architecture treats the collator/factory
  boundary as the one place mode-specific shapes are allowed to surface.
  The caller picks the factory matching `cfg.mode`.

- **`build_dataloader(...)`** — legacy SFT-only signature kept for old
  call sites; internally forces `mode = SFT`, reloads via the unified
  path, and delegates to `build_sft_dataloader`.

The trailing comment block is a plain-text pipeline diagram tracing data
from `tokenize_dataset.py` output through to the training loop.

---

## `rollout_buffer.h` — In-memory PPO/GRPO buffer (new component)

This is not a disk loader — it's the missing piece that sits between the
rollout phase and the PPO/GRPO update phase.

- **`RolloutEntry`** — what the training loop pushes after one rollout
  batch: `prompt_ids`, `response_ids`, `old_log_probs`, `ref_log_probs`,
  `rewards`, `advantages`. For GRPO, the training loop has already
  expanded the batch dimension to `B×G` before pushing — the buffer
  doesn't know or care.

- **`RolloutMiniBatch`** — what comes back during the update phase, same
  fields, sliced to minibatch size.

- **`RolloutBuffer` lifecycle:**
  1. **`push(entry)`** — called repeatedly during rollout collection;
     entries accumulate in `entries_`.
  2. **`seal()`** — concatenates all pushed entries along dim 0 into flat
     tensors, optionally normalizes `advantages` to zero-mean/unit-std
     (standard PPO practice), records `total_samples_`, and generates a
     random permutation `perm_` for shuffled iteration.
  3. **`next_minibatch()`** — returns slices of size `minibatch_size_`
     according to `perm_`, advancing `cursor_`; returns `nullopt` when
     the epoch over the buffer is exhausted.
  4. **`reset_cursor()`** — re-shuffles and resets `cursor_` to 0, used
     to run multiple PPO epochs over the same sealed rollout data.
  5. **`clear()`** — wipes everything, ready for the next rollout
     collection round.

`num_minibatches()` and `total_samples()` are convenience accessors for
logging/loop bounds.

---

## `distributed.h` — Multi-GPU bootstrap (mode-agnostic)

- **`DistributedContext`** — holds `rank`, `world_size`, `local_rank`,
  `device`.

- **`init_distributed(backend)`**:
  - If `RANK` env var isn't set → single-process fallback: rank 0,
    world_size 1, device = CUDA:0 if available else CPU.
  - If set → reads `RANK`/`WORLD_SIZE`/`LOCAL_RANK` from env, assigns
    `device = cuda:local_rank`, and (only if compiled with
    `USE_DISTRIBUTED`) sets up a `TCPStore` + `ProcessGroupNCCL` for
    actual multi-GPU collective communication.

- **`cleanup_distributed()` / `barrier(ctx)`** — teardown and
  synchronization, no-ops in single-process mode.

This file never branches on `TrainingMode` — it answers "which
device/rank am I" regardless of what data is being loaded.

---

## `tokenize_dataset.py` — Offline preprocessing

Converts JSONL records into the `.bin` binary format consumed by
`BinarySequenceReader`.

- **`load_tokenizer(name)`** — tries to load a HuggingFace
  `AutoTokenizer`; falls back to a trivial `WhitespaceTokenizer`
  (hash-based ids) if unavailable, useful for testing the binary pipeline
  without downloading model weights.

- **`write_binary(path, sequences, max_len)`** — writes the exact format
  the C++ reader expects: `uint32 count`, then per sequence `uint16
  length` + `uint32[length]` tokens, truncating to `max_len` (and to the
  hard `uint16` cap of 65535) if needed.

- **Mode handlers:**
  - `run_sft` — expects `{"prompt", "response"}` per line, writes
    `prompts.bin` + `responses.bin`.
  - `run_preference` — expects `{"prompt", "chosen", "rejected"}`, writes
    `prompts.bin` + `chosen.bin` + `rejected.bin` (shared by
    `reward_model` and `dpo` modes since the on-disk format is identical
    — they only diverge in the C++ collator).
  - `run_rollout` — expects `{"prompt"}` only, writes `prompts.bin`,
    truncated to `max_prompt_len`.

- **CLI args** — `--input`, `--output-dir`, `--mode
  {sft,reward_model,dpo,rollout}`, `--tokenizer`, `--max-seq-len`,
  `--max-prompt-len`.

---

## `main.cpp` — End-to-end example

One `run_*` function per mode, each following the same pattern: load
samples via `UnifiedDatasetReader::load(cfg)`, wrap in `UnifiedDataset`,
build the matching prefetcher via `dataloader.h`, then pull a few batches
and print their tensor shapes.

`run_rollout` additionally shows the PPO/GRPO training-loop sketch: after
getting a `RolloutBatch` (just `prompt_ids`/`prompt_mask`), it constructs
a `RolloutEntry` with placeholder zero tensors for everything that would
normally come from generation/reward/advantage computation, pushes it
into a `RolloutBuffer`, then calls `seal()` and prints the resulting
minibatch count — illustrating exactly where the data loader's
responsibility ends and the training loop's begins.

`main()` parses the mode from `argv[1]`, sets up a default
`DataLoaderConfig` (batch size 4, `max_prompt_len=128`, `max_gen_len=256`,
`grpo_group_size=8`), calls `init_distributed()`, dispatches to the right
`run_*`, and calls `cleanup_distributed()` at the end.

---

## `CMakeLists.txt` — Build configuration

Standard libtorch build: requires CMake 3.18+, C++17,
`find_package(Torch REQUIRED)` (pointed at libtorch via
`CMAKE_PREFIX_PATH`), builds `dataloader_demo` from `main.cpp`, links
`TORCH_LIBRARIES` and `Threads` (needed by the prefetcher's thread pool),
with an optional `USE_DISTRIBUTED` flag that defines the macro gating the
NCCL code paths in `distributed.h`.