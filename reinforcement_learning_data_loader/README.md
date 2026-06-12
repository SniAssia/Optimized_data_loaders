# Unified Post-Training Data Loader (SFT / Reward Model / DPO / PPO-GRPO Rollout)

A single C++ data loader that switches between four training modes via
`DataLoaderConfig::mode`:

| Mode           | Files read                              | Batch struct       | Labels |
|----------------|------------------------------------------|--------------------|--------|
| `SFT`          | `prompts.bin`, `responses.bin`            | `SFTBatch`         | yes (response only) |
| `REWARD_MODEL` | `prompts.bin`, `chosen.bin`, `rejected.bin` | `RewardModelBatch` | no |
| `DPO`          | `prompts.bin`, `chosen.bin`, `rejected.bin` | `DPOBatch`         | yes (both sides) |
| `ROLLOUT`      | `prompts.bin` only                        | `RolloutBatch`     | n/a (prompt-only) |

## Architecture

```
tokenize_dataset.py --mode {sft|reward_model|dpo|rollout}
        │
        ▼
UnifiedDatasetReader::load(cfg)      reads only the files the mode needs
        │
        ▼
UnifiedDataset(samples, cfg)         mode-specific truncation + lengths()
        │
        ▼
shard_indices(...)                   mode-agnostic, multi-GPU sharding
        │
        ▼
BucketBatchSampler(lengths, ...)     mode-agnostic, length-bucketed batches
        │
        ▼
UnifiedCollator::collate_*(items)    mode-specific batch struct
        │
        ▼
TypedPrefetcher<BatchT>              background workers -> GPU
        │
        ▼
Training loop
```

Only `dataset.h` (raw sample struct, truncation, config) and `collator.h`
(batch structs, collation logic) are mode-aware. `sampler.h`,
`distributed.h`, and the core of `prefetcher.h` are completely generic.

## Files

- `dataset.h` — `RawSample`, `DataLoaderConfig`, `UnifiedDatasetReader`,
  `UnifiedDataset` (+ legacy `InstructionDataset`/`BinaryDatasetReader`
  aliases for old SFT-only code).
- `collator.h` — `SFTBatch`, `RewardModelBatch`, `DPOBatch`, `RolloutBatch`,
  `UnifiedCollator`.
- `sampler.h` — `shard_indices`, `BucketBatchSampler` (unchanged, generic).
- `prefetcher.h` — `TypedPrefetcher<BatchT>`, with concrete aliases
  `SFTPrefetcher`, `RewardModelPrefetcher`, `DPOPrefetcher`,
  `RolloutPrefetcher` (+ legacy `CUDAPrefetcher` alias).
- `dataloader.h` — one `build_*_dataloader()` factory per mode, wiring
  dataset → sampler → collator → prefetcher.
- `rollout_buffer.h` — `RolloutBuffer`, the in-memory store used by the
  PPO/GRPO update phase (push rollouts → seal → iterate minibatches → clear).
- `distributed.h` — `init_distributed`, `DistributedContext`, `barrier`.
- `tokenize_dataset.py` — produces the `.bin` files for each mode from
  a JSONL dataset.
- `main.cpp` / `CMakeLists.txt` — runnable example for all four modes.

## Quick start

```bash
# 1. Tokenize your data (whitespace tokenizer for a quick smoke test,
#    or pass --tokenizer <hf-model-name> for a real tokenizer)
python3 tokenize_dataset.py --input sft_data.jsonl  --output-dir ./data --mode sft
python3 tokenize_dataset.py --input pref_data.jsonl --output-dir ./data --mode reward_model
python3 tokenize_dataset.py --input pref_data.jsonl --output-dir ./data --mode dpo
python3 tokenize_dataset.py --input prompts.jsonl   --output-dir ./data --mode rollout

# 2. Build
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . -j

# 3. Run (from inside ./data, or pass paths via DataLoaderConfig)
./dataloader_demo sft
./dataloader_demo reward_model
./dataloader_demo dpo
./dataloader_demo rollout
```

## How PPO / GRPO use this loader

The data loader is involved at most **three times** per PPO/GRPO pipeline:

1. **SFT** — warm-up training run, `Mode::SFT`.
2. **Reward model training** — only if you don't have a verifiable
   reward signal, `Mode::REWARD_MODEL`.
3. **Rollout** — `Mode::ROLLOUT`, delivers prompt-only batches
   (left-padded, with `max_gen_len` reserved generation columns).
   `cfg.grpo_group_size` is carried in the config purely for the
   training loop's benefit — the data loader never reads it.

Everything after step 3 (generation, KL, reward scoring, advantage
computation, PPO/GRPO update epochs) happens in the training loop using
`RolloutBuffer`, which is **not** a disk-reading data loader — it's an
in-memory tensor store:

```cpp
RolloutBuffer buffer(/*minibatch_size=*/cfg.batch_size);

while (auto prompt_batch = rollout_prefetcher->next()) {
    // ... actor generates responses into prompt_batch->prompt_ids ...
    // ... reference model log-probs, reward model scores, advantages ...
    buffer.push(RolloutEntry{ ... });
}

buffer.seal();
for (int epoch = 0; epoch < K; ++epoch) {
    buffer.reset_cursor();
    while (auto mb = buffer.next_minibatch()) {
        // PPO/GRPO gradient step
    }
}
buffer.clear();
```

## DPO vs Reward Model

Both read the same `(prompt, chosen, rejected)` triplet files. The only
difference is in `UnifiedCollator`:

- `collate_reward_model` → `RewardModelBatch` (no labels; the reward model
  reads only the last-token hidden state).
- `collate_dpo` → `DPOBatch` (adds `chosen_labels`/`rejected_labels` with
  the prompt portion masked as `-100`, since DPO needs token-level log
  probabilities).
