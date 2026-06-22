#!/usr/bin/env python3
import argparse
import json
import os
import queue
import random
import struct
import threading

MAX_SEQ_LEN_HARD_CAP= 65535
DEFAULT_TOKENIZER = "inceptionai/jais-family-590m"
DEFAULT_SHARD_SIZE = 10_000
DEFAULT_SHUFFLE_BUF = 50_000

# Sentinel — pushed by SourceThread when its source is exhausted
_DONE = object()


def load_tokenizer(name_or_path):
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer '{name_or_path}': {e}\n"
            "Make sure transformers is installed and you have network access."
        ) from e
    return HFTokenizerWrapper(tok)


class HFTokenizerWrapper:
    def __init__(self, tok):
        self.tok = tok
        self.eos_token_id = tok.eos_token_id if tok.eos_token_id is not None else 2

    def encode(self, text, add_eos=True):
        ids = self.tok.encode(text, add_special_tokens=False)
        if add_eos and self.eos_token_id is not None:
            ids = ids + [self.eos_token_id]
        return ids


class ShardWriter:
    """
    Accumulates token ID lists for one shard in RAM.
    flush() writes all fields to disk and resets.
    Peak RAM = one shard worth of token IDs (~20 MB at shard_size=10000).
    """

    def __init__(self, out_dir, max_len):
        self.out_dir       = out_dir
        self.max_len       = max_len
        self._buffers      = {}
        self._count        = 0
        self._n_truncated  = 0

    def add(self, field, token_ids):
        if self.max_len and len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
            self._n_truncated += 1
        if len(token_ids) > MAX_SEQ_LEN_HARD_CAP:
            token_ids = token_ids[:MAX_SEQ_LEN_HARD_CAP]
            self._n_truncated += 1
        self._buffers.setdefault(field, []).append(token_ids)

    def commit_record(self):
        self._count += 1

    def n_records(self):
        return self._count

    def snapshot_and_reset(self):
        """
        Returns (buffers, count, n_truncated) and resets internal state.
        Called by the main thread to hand data off to ShardFlusher
        without copying — the flusher owns the snapshot.
        """
        snap = (self._buffers, self._count, self._n_truncated)
        self._buffers     = {}
        self._count       = 0
        self._n_truncated = 0
        return snap

    def flush_to_disk(self, out_dir, buffers, count, n_truncated):
        """
        Writes a snapshot (produced by snapshot_and_reset) to disk.
        Called from the ShardFlusher background thread.
        """
        if count == 0:
            return 0

        os.makedirs(out_dir, exist_ok=True)

        for field, seqs in buffers.items():
            path = os.path.join(out_dir, f"{field}.bin")
            with open(path, "wb") as f:
                f.write(struct.pack("<I", len(seqs)))
                for seq in seqs:
                    f.write(struct.pack("<H", len(seq)))
                    if seq:
                        f.write(struct.pack(f"<{len(seq)}I", *seq))
            print(f"  [shard] wrote {len(seqs)} sequences → {path}"
                  + (f" ({n_truncated} truncated)" if n_truncated else ""))
        return count
# Stage 1 — SourceThread
# One thread per input source. Streams records and pushes them into a
# shared queue. Pushes _DONE when the source is exhausted.

class SourceThread:
    """
    Wraps one record generator in a background thread.
    Records are pushed into shared_queue as plain dicts.
    When the generator is exhausted, pushes _DONE once.
    """

    def __init__(self, generator, shared_queue, source_name=""):
        self.generator    = generator
        self.queue        = shared_queue
        self.source_name  = source_name
        self._thread      = threading.Thread(
            target=self._run, daemon=True,
            name=f"source-{source_name}"
        )

    def start(self):
        self._thread.start()

    def join(self):
        self._thread.join()

    def _run(self):
        n = 0
        try:
            for rec in self.generator:
                self.queue.put(rec)
                n += 1
        except Exception as e:
            print(f"[SourceThread:{self.source_name}] ERROR: {e}")
        finally:
            self.queue.put(_DONE)
            print(f"[SourceThread:{self.source_name}] done — pushed {n} records")
# Stage 2 — ShuffleBuffer
# Reads from the shared queue filled by all SourceThreads.
# Maintains a pool of N text records from all sources mixed together.
# Randomly evicts one record at a time downstream.
# Larger buffer = better shuffle quality, more RAM used for text (~25MB
# at 50000 records since text records are only ~500 bytes each).
class ShuffleBuffer:
    """
    Approximate global shuffle across all source streams.

    Internally keeps a list (the pool) of up to `size` records.
    - While pool is not full: accumulate records from the queue.
    - Once full: for each new record arriving, randomly pick and
      yield one record from the pool, replace it with the new record.
    - When all sources are done: yield remaining pool in random order.

    This is identical to WebDataset's shuffle buffer logic.
    """

    def __init__(self, shared_queue, n_sources, size=DEFAULT_SHUFFLE_BUF):
        self.queue     = shared_queue
        self.n_sources = n_sources
        self.size      = size

    def __iter__(self):
        pool         = []
        done_count   = 0   # how many SourceThreads have sent _DONE

        while done_count < self.n_sources or pool:

            # Drain queue into pool until full or all sources done
            while len(pool) < self.size and done_count < self.n_sources:
                item = self.queue.get()
                if item is _DONE:
                    done_count += 1
                    print(f"[ShuffleBuffer] source finished "
                          f"({done_count}/{self.n_sources} done), "
                          f"pool size={len(pool)}")
                else:
                    pool.append(item)

            if not pool:
                break

            # Yield one random record from the pool
            idx        = random.randrange(len(pool))
            # Swap with last element for O(1) removal
            pool[idx], pool[-1] = pool[-1], pool[idx]
            yield pool.pop()

        # All sources exhausted — drain remaining pool in random order
        random.shuffle(pool)
        yield from pool
        print(f"[ShuffleBuffer] fully drained")

# Stage 3 — ShardFlusher
# Receives full shard snapshots from the main thread via an internal queue.
# Writes .bin files to disk in a background thread.
# Main thread never waits for disk I/O.
class ShardFlusher:
    """
    Background thread that writes shard snapshots to disk.

    Main thread calls submit(shard_dir, snapshot) — returns immediately.
    Background thread picks up the snapshot and writes .bin files.
    Call join() at the end to wait for all pending writes to complete.
    """

    def __init__(self, writer):
        self._writer  = writer       # ShardWriter instance (used only for flush_to_disk)
        self._queue   = queue.Queue()
        self._thread  = threading.Thread(
            target=self._run, daemon=True, name="shard-flusher"
        )
        self._thread.start()
        self._manifest_entries = []  # collected here, thread-safe via GIL on list.append
        self._lock = threading.Lock()

    def submit(self, shard_dir, snapshot, shard_idx):
        """
        Hand off a full shard snapshot to the background thread.
        snapshot = (buffers, count, n_truncated) from ShardWriter.snapshot_and_reset()
        Returns immediately — does not wait for disk write.
        """
        self._queue.put((shard_dir, snapshot, shard_idx))

    def join(self):
        #Wait for all pending shard writes to complete
        self._queue.put(None)   # sentinel — tells background thread to exit
        self._thread.join()

    def manifest_entries(self):
        return self._manifest_entries

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                break   # sentinel received — exit
            shard_dir, (buffers, count, n_truncated), shard_idx = item
            n = self._writer.flush_to_disk(shard_dir, buffers, count, n_truncated)
            with self._lock:
                self._manifest_entries.append({
                    "shard":     shard_idx,
                    "n_samples": n,
                    "dir":       shard_dir,
                })
            print(f"[ShardFlusher] shard_{shard_idx:02d} written ({n} records)")


def _make_shard_dir(base_out_dir, shard_idx, use_sharding):
    if use_sharding:
        return os.path.join(base_out_dir, f"shard_{shard_idx:02d}")
    return base_out_dir


def _write_manifest(out_dir, mode, manifest, use_sharding):
    if not use_sharding:
        return
    # Sort by shard index so the manifest is ordered
    manifest = sorted(manifest, key=lambda e: e["shard"])
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"mode": mode, "shards": manifest}, f, indent=2)
    print(f"[tokenize_dataset] manifest written → {path}")
# Streaming tokenization modes
# These now receive a ShuffleBuffer iterator instead of a plain generator.
# Internally they use ShardFlusher for async disk writes.
# Logic is otherwise identical to the previous version.
def run_sft(record_gen, tok, out_dir, max_len, shard_size):
    use_sharding = shard_size > 0
    shard_idx    = 0
    writer       = ShardWriter(_make_shard_dir(out_dir, shard_idx, use_sharding), max_len)
    flusher      = ShardFlusher(writer)

    for rec in record_gen:
        writer.add("prompts",   tok.encode(rec["prompt"],   add_eos=False))
        writer.add("responses", tok.encode(rec["response"], add_eos=True))
        writer.commit_record()

        if use_sharding and writer.n_records() >= shard_size:
            snap   = writer.snapshot_and_reset()
            shard_dir= _make_shard_dir(out_dir, shard_idx, use_sharding)
            flusher.submit(shard_dir, snap, shard_idx)
            shard_idx+= 1
            writer.out_dir = _make_shard_dir(out_dir, shard_idx, use_sharding)

    # flush final partial shard
    if writer.n_records() > 0:
        snap      = writer.snapshot_and_reset()
        shard_dir = _make_shard_dir(out_dir, shard_idx, use_sharding)
        flusher.submit(shard_dir, snap, shard_idx)

    flusher.join()
    _write_manifest(out_dir, "sft", flusher.manifest_entries(), use_sharding)
    total = sum(e["n_samples"] for e in flusher.manifest_entries())
    print(f"[tokenize_dataset] sft: {total} records in "
          f"{len(flusher.manifest_entries())} shard(s) → {out_dir}")


def run_preference(record_gen, tok, out_dir, max_len, shard_size, label):
    use_sharding = shard_size > 0
    shard_idx    = 0
    writer       = ShardWriter(_make_shard_dir(out_dir, shard_idx, use_sharding), max_len)
    flusher      = ShardFlusher(writer)

    for rec in record_gen:
        writer.add("prompts",tok.encode(rec["prompt"],   add_eos=False))
        writer.add("chosen",         tok.encode(rec["chosen"],   add_eos=True))
        writer.add("rejected", tok.encode(rec["rejected"], add_eos=True))
        writer.commit_record()

        if use_sharding and writer.n_records() >= shard_size:
            snap      = writer.snapshot_and_reset()
            shard_dir = _make_shard_dir(out_dir, shard_idx, use_sharding)
            flusher.submit(shard_dir, snap, shard_idx)
            shard_idx += 1
            writer.out_dir = _make_shard_dir(out_dir, shard_idx, use_sharding)

    if writer.n_records() > 0:
        snap      = writer.snapshot_and_reset()
        shard_dir = _make_shard_dir(out_dir, shard_idx, use_sharding)
        flusher.submit(shard_dir, snap, shard_idx)

    flusher.join()
    _write_manifest(out_dir, label, flusher.manifest_entries(), use_sharding)
    total = sum(e["n_samples"] for e in flusher.manifest_entries())
    print(f"[tokenize_dataset] {label}: {total} records in "
          f"{len(flusher.manifest_entries())} shard(s) → {out_dir}")


def run_rollout(record_gen, tok, out_dir, max_prompt_len, shard_size):
    use_sharding = shard_size > 0
    shard_idx    = 0
    writer       = ShardWriter(_make_shard_dir(out_dir, shard_idx, use_sharding),
                               max_prompt_len)
    flusher      = ShardFlusher(writer)

    for rec in record_gen:
        writer.add("prompts", tok.encode(rec["prompt"], add_eos=False))
        writer.commit_record()

        if use_sharding and writer.n_records() >= shard_size:
            snap      = writer.snapshot_and_reset()
            shard_dir = _make_shard_dir(out_dir, shard_idx, use_sharding)
            flusher.submit(shard_dir, snap, shard_idx)
            shard_idx += 1
            writer.out_dir = _make_shard_dir(out_dir, shard_idx, use_sharding)

    if writer.n_records() > 0:
        snap      = writer.snapshot_and_reset()
        shard_dir = _make_shard_dir(out_dir, shard_idx, use_sharding)
        flusher.submit(shard_dir, snap, shard_idx)

    flusher.join()
    _write_manifest(out_dir, "rollout", flusher.manifest_entries(), use_sharding)
    total = sum(e["n_samples"] for e in flusher.manifest_entries())
    print(f"[tokenize_dataset] rollout: {total} prompts in "
          f"{len(flusher.manifest_entries())} shard(s) → {out_dir}")


def _resolve_input(input_arg, split):
    import prepare_dataset as pd
    is_local = input_arg.endswith(".jsonl") or os.path.isfile(input_arg)
    if is_local:
        print(f"[tokenize_dataset] input: local JSONL → {input_arg}")
        return pd.stream_jsonl(input_arg)
    else:
        print(f"[tokenize_dataset] input: HuggingFace → {input_arg} (split={split})")
        return pd.stream_dataset(input_arg, split=split)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
    "--input", required=True, action="append",
    help="..."
    )
    parser.add_argument(
        "--split", default="train", help="HuggingFace split (ignored for local files, default: train)" )
    parser.add_argument(
        "--output-dir", required=True,
        help="Root directory to write shard subdirectories into")
    parser.add_argument(
        "--mode", required=True,
        choices=["sft", "reward_model", "dpo", "rollout"],
        help="Which binary files to produce" )
    parser.add_argument(
        "--tokenizer", default=DEFAULT_TOKENIZER,
        help=f"HuggingFace tokenizer name/path (default: {DEFAULT_TOKENIZER})"  )
    parser.add_argument(
        "--max-seq-len", type=int, default=2048,
        help="Max token length for sft/reward_model/dpo sequences"  )
    parser.add_argument(
        "--max-prompt-len", type=int, default=512,
        help="Max prompt token length for rollout mode" )
    parser.add_argument(
        "--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
        help="Records per shard (0 = flat layout, backward compatible)" )
    # NEW flag
    parser.add_argument(
        "--shuffle-buffer-size", type=int, default=DEFAULT_SHUFFLE_BUF,
        help="Shuffle buffer size in records across all sources "
             "(larger = better shuffle quality, more RAM for text, "
             "default: 50000 ≈ 25 MB)"
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # ---- load tokenizer once ----
    print(f"[tokenize_dataset] loading tokenizer '{args.tokenizer}' ...")
    tok = load_tokenizer(args.tokenizer)
    print(f"[tokenize_dataset] tokenizer loaded (eos_token_id={tok.eos_token_id})")
    # ---- Stage 1: build one SourceThread per input ----
    shared_queue = queue.Queue(maxsize=args.shuffle_buffer_size * 2)
    threads = []
    for inp in args.input:
        gen    = _resolve_input(inp, args.split)
        thread = SourceThread(gen, shared_queue, source_name=inp)
        threads.append(thread)

    print(f"[tokenize_dataset] starting {len(threads)} source thread(s) ...")
    for t in threads:
        t.start()

    # ---- Stage 2: shuffle buffer merges all streams ----
    shuffle_buf = ShuffleBuffer(
        shared_queue,
        n_sources=len(threads),
        size=args.shuffle_buffer_size,
    )

    #  tokenize + async shard flusher ----
    # shuffle_buf is the record_gen passed to run_* functions
    # ShardFlusher inside each run_* handles async disk writes

    if args.mode == "sft":
        run_sft(shuffle_buf, tok, args.output_dir,
                args.max_seq_len, args.shard_size)

    elif args.mode in ("reward_model", "dpo"):
        run_preference(shuffle_buf, tok, args.output_dir,
                       args.max_seq_len, args.shard_size, args.mode)

    elif args.mode == "rollout":
        run_rollout(shuffle_buf, tok, args.output_dir,
                    args.max_prompt_len, args.shard_size)
    for t in threads:
        t.join()

    print(f"[tokenize_dataset] all done → {args.output_dir}")


if __name__ == "__main__":
    main()









""" 
How to Use It
Single source — identical to before:
bashpython3 tokenize_dataset.py \
    --input Anthropic/hh-rlhf \
    --output-dir out/ \
    --mode reward_model
Multiple sources — new:
bashpython3 tokenize_dataset.py \
    --input Anthropic/hh-rlhf \
    --input HuggingFaceH4/ultrafeedback_binarized \
    --input my_local_data.jsonl \
    --output-dir out/ \
    --mode reward_model \
    --shuffle-buffer-size 50000
"""


"""
tokenize_dataset.py — Streaming multi-source tokenization + sharding
for the post-training data loader (SFT / REWARD_MODEL / DPO / ROLLOUT).

CHANGES FROM PREVIOUS VERSION:
    - Accepts multiple --input sources simultaneously
    - Stage 1: one SourceThread per input, all stream in parallel
    - Stage 2: ShuffleBuffer merges all streams + approximate global shuffle
    - Stage 3: ShardFlusher writes .bin files asynchronously in background
    - Main pipeline never blocks on disk I/O
    - New flag: --shuffle-buffer-size (default 50000)
    - All previous flags preserved for backward compatibility

Single-source usage (identical to before):
    python3 tokenize_dataset.py \
        --input Anthropic/hh-rlhf \
        --output-dir out/ \
        --mode reward_model

Multi-source usage (new):
    python3 tokenize_dataset.py \
        --input Anthropic/hh-rlhf \
        --input HuggingFaceH4/ultrafeedback_binarized \
        --input my_local_data.jsonl \
        --output-dir out/ \
        --mode reward_model \
        --shuffle-buffer-size 50000

Binary shard format (unchanged):
    uint32  n_samples
    [per sample]
        uint16  length
        uint32  token_ids[length]

Output layout (unchanged):
    <output-dir>/
        shard_00/prompts.bin  chosen.bin  rejected.bin
        shard_01/...
        manifest.json
"""
