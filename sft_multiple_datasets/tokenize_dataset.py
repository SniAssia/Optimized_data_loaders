#!/usr/bin/env python3
"""
tokenize_dataset.py — Multi-source streaming tokenization with sharded output.

Architecture : 
Single dataset:
    load_dataset (streaming) ──► tokenize ──► ShardWriter ──► ShardFlusher ──► disk

Multiple datasets:
    SourceThread_0 ──┐
    SourceThread_1 ──┼──► shared Queue ──► ShuffleBuffer ──► tokenize ──► ShardWriter ──► ShardFlusher ──► disk
    SourceThread_N ──┘

Each shard on disk contains:
    shard_XX/
        samples.bin     ← binary: n_samples, then per sample: (prompt_len, response_len, prompt_ids..., response_ids..., attention_mask...)
        lengths.bin     ← binary: n_samples × (uint16 prompt_len, uint16 response_len)

manifest.json:
    { "max_seq_len": N, "pad_token_id": N, "shards": [ {"shard": N, "n_samples": N, "dir": "..."}, ... ] }

Binary sample format (samples.bin):
    uint32  n_samples
    per sample:
        uint16  prompt_len
        uint16  response_len
        uint32  prompt_ids    [max_seq_len]   ← always full max_seq_len (padded)
        uint32  response_ids  [max_seq_len]   ← always full max_seq_len (padded)
        uint8   attention_mask[max_seq_len]   ← 1 for real tokens, 0 for padding
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import struct
import threading
from typing import List, Optional, Tuple

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

# ── constants ────────────────────────────────────────────────────────────────
DEFAULT_TOKENIZER   = "inceptionai/jais-family-590m"
DEFAULT_SHARD_SIZE  = 10_000
DEFAULT_SHUFFLE_BUF = 50_000
MAX_SEQ_LEN_HARD    = 65535

_DONE = object()   # sentinel pushed by SourceThread when exhausted

# column detection (kept from original) 
INSTRUCTION_CANDIDATES = ["prompt", "instruction", "input", "question",
                           "human", "user", "context", "query"]
RESPONSE_CANDIDATES    = ["completion", "response", "output", "answer",
                           "assistant", "gpt", "text", "target"]


def detect_columns(row: dict) -> Optional[Tuple[str, str]]:
    keys      = [k.lower() for k in row.keys()]
    key_map   = {k.lower(): k for k in row.keys()}
    inst_col  = next((key_map[c] for c in INSTRUCTION_CANDIDATES if c in keys), None)
    resp_col  = next((key_map[c] for c in RESPONSE_CANDIDATES    if c in keys), None)
    return (inst_col, resp_col) if inst_col and resp_col else None


def extract_from_messages(row: dict) -> Optional[Tuple[str, str]]:
    for col in ["messages", "conversations", "conversation", "dialogue", "chat"]:
        msgs = row.get(col)
        if not isinstance(msgs, list) or len(msgs) < 2:
            continue
        instruction = response = None
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            role    = msg.get("role", msg.get("from", "")).lower().strip()
            content = msg.get("content", msg.get("value", "")).strip()
            if not content:
                continue
            if role in ("user", "human", "instruction") and instruction is None:
                instruction = content
            elif role in ("assistant", "gpt", "bot") and response is None:
                response = content
        if instruction and response:
            return instruction, response
    return None


def extract_pair(row: dict, inst_col=None, resp_col=None) -> Tuple[str, str]:
    if inst_col and resp_col:
        i = str(row.get(inst_col, "")).strip()
        r = str(row.get(resp_col, "")).strip()
        if i and r:
            return i, r
    result = extract_from_messages(row)
    if result:
        return result
    if "instruction" in row and "output" in row:
        i = str(row.get("instruction", "")).strip()
        extra = str(row.get("input", "")).strip()
        r = str(row.get("output", "")).strip()
        if extra:
            i = f"{i}\n{extra}"
        if i and r:
            return i, r
    detected = detect_columns(row)
    if detected:
        i_col, r_col = detected
        i = str(row.get(i_col, "")).strip()
        r = str(row.get(r_col, "")).strip()
        if i and r:
            return i, r
    raise ValueError(f"Cannot extract pair. Columns: {list(row.keys())}")


# ── tokenizer wrapper ────────────────────────────────────────────────────────
class HFTokenizerWrapper:
    def __init__(self, tok):
        self.tok         = tok
        self.eos_id      = tok.eos_token_id if tok.eos_token_id is not None else 2
        self.pad_id      = tok.pad_token_id if tok.pad_token_id is not None else 0

    def encode(self, text: str) -> List[int]:
        ids = self.tok.encode(text, add_special_tokens=True,
                              truncation=False)
        return ids


def load_tokenizer(name: str) -> HFTokenizerWrapper:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return HFTokenizerWrapper(tok)

class TokenizedSample:
    """
    A single sample — variable length, no padding.
    prompt_ids and response_ids are stored at their real token count.
    Padding happens at batch construction time in the C++ collator.
    """
    __slots__ = ("prompt_ids", "response_ids", "prompt_len", "response_len")

    def __init__(self, prompt_ids: List[int], response_ids: List[int],
                 prompt_len: int, response_len: int):
        self.prompt_ids   = prompt_ids
        self.response_ids = response_ids
        self.prompt_len   = prompt_len
        self.response_len = response_len
        
# ── shard writer ─────────────────────────────────────────────────────────────
class ShardWriter:
    """
    Accumulates TokenizedSample objects in RAM.
    snapshot_and_reset() hands off the buffer to ShardFlusher without copying.
    """

    def __init__(self, max_seq_len: int):
        self.max_seq_len  = max_seq_len
        self._samples: List[TokenizedSample] = []
        self._n_truncated = 0

    def add(self, sample: TokenizedSample, was_truncated: bool):
        if was_truncated:
            self._n_truncated += 1
        self._samples.append(sample)

    def n_records(self) -> int:
        return len(self._samples)

    def snapshot_and_reset(self):
        snap = (self._samples, self._n_truncated)
        self._samples     = []
        self._n_truncated = 0
        return snap

    @staticmethod
    def flush_to_disk(shard_dir: str, samples: List[TokenizedSample],
                      n_truncated: int, max_seq_len: int):
        if not samples:
            return 0
        os.makedirs(shard_dir, exist_ok=True)

        n = len(samples)

        # ── samples.bin ──────────────────────────────────────────────
        samples_path = os.path.join(shard_dir, "samples.bin")
        with open(samples_path, "wb") as f:
            f.write(struct.pack("<I", n))
            for s in samples:
                f.write(struct.pack("<H", s.prompt_len))
                f.write(struct.pack(f"<{s.prompt_len}I", *s.prompt_ids))   # exact length
                f.write(struct.pack("<H", s.response_len))
                f.write(struct.pack(f"<{s.response_len}I", *s.response_ids)) # exact length

        # ── lengths.bin ──────────────────────────────────────────────
        lengths_path = os.path.join(shard_dir, "lengths.bin")
        with open(lengths_path, "wb") as f:
            f.write(struct.pack("<I", n))
            for s in samples:
                f.write(struct.pack("<H", s.prompt_len))
                f.write(struct.pack("<H", s.response_len))

        trunc_msg = f" ({n_truncated} truncated)" if n_truncated else ""
        print(f"  [shard] wrote {n} samples → {shard_dir}{trunc_msg}")
        return n


# ── shard flusher ─────────────────────────────────────────────────────────────
class ShardFlusher:
    """Background thread that writes shard snapshots to disk."""

    def __init__(self, max_seq_len: int):
        self._max_seq_len      = max_seq_len
        self._queue            = queue.Queue()
        self._manifest_entries = []
        self._lock             = threading.Lock()
        self._thread           = threading.Thread(target=self._run,
                                                   daemon=True,
                                                   name="shard-flusher")
        self._thread.start()

    def submit(self, shard_dir: str, snapshot, shard_idx: int):
        self._queue.put((shard_dir, snapshot, shard_idx))

    def join(self):
        self._queue.put(None)
        self._thread.join()

    def manifest_entries(self):
        return self._manifest_entries

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            shard_dir, (samples, n_truncated), shard_idx = item
            n = ShardWriter.flush_to_disk(shard_dir, samples,
                                          n_truncated, self._max_seq_len)
            with self._lock:
                self._manifest_entries.append({
                    "shard":     shard_idx,
                    "n_samples": n,
                    "dir":       shard_dir,
                })
            print(f"[ShardFlusher] shard_{shard_idx:02d} written ({n} records)")


# ── source thread ─────────────────────────────────────────────────────────────
class SourceThread:
    """Streams records from one generator into a shared queue."""

    def __init__(self, generator, shared_queue: queue.Queue, name: str = ""):
        self.generator = generator
        self.queue     = shared_queue
        self.name      = name
        self._thread   = threading.Thread(target=self._run, daemon=True,
                                           name=f"source-{name}")

    def start(self):  self._thread.start()
    def join(self):   self._thread.join()

    def _run(self):
        n = 0
        try:
            for rec in self.generator:
                self.queue.put(rec)
                n += 1
        except Exception as e:
            print(f"[SourceThread:{self.name}] ERROR: {e}")
        finally:
            self.queue.put(_DONE)
            print(f"[SourceThread:{self.name}] done — pushed {n} records")


# ── shuffle buffer ────────────────────────────────────────────────────────────
class ShuffleBuffer:
    """Approximate global shuffle across all source streams."""

    def __init__(self, shared_queue: queue.Queue, n_sources: int,
                 size: int = DEFAULT_SHUFFLE_BUF):
        self.queue     = shared_queue
        self.n_sources = n_sources
        self.size      = size

    def __iter__(self):
        pool       = []
        done_count = 0

        while done_count < self.n_sources or pool:
            while len(pool) < self.size and done_count < self.n_sources:
                item = self.queue.get()
                if item is _DONE:
                    done_count += 1
                    print(f"[ShuffleBuffer] source finished "
                          f"({done_count}/{self.n_sources}), pool={len(pool)}")
                else:
                    pool.append(item)

            if not pool:
                break

            idx          = random.randrange(len(pool))
            pool[idx], pool[-1] = pool[-1], pool[idx]
            yield pool.pop()

        random.shuffle(pool)
        yield from pool
        print("[ShuffleBuffer] fully drained")


# ── dataset streaming ─────────────────────────────────────────────────────────
def stream_hf_dataset(dataset_name: str, split: str,
                      inst_col=None, resp_col=None):
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split,
                      streaming=True, trust_remote_code=True)
    for row in ds:
        try:
            i, r = extract_pair(row, inst_col, resp_col)
            if i and r:
                yield {"instruction": i, "response": r}
        except ValueError:
            continue


def stream_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            try:
                i, r = extract_pair(rec)
                if i and r:
                    yield {"instruction": i, "response": r}
            except ValueError:
                continue


def tokenize_record(rec, tok, max_seq_len):
    prompt_ids = tok.encode(rec["instruction"])
    resp_ids   = tok.encode(rec["response"])

    was_truncated = False

    if not prompt_ids or prompt_ids[-1] != tok.eos_id:
        prompt_ids.append(tok.eos_id)
    if not resp_ids or resp_ids[-1] != tok.eos_id:
        resp_ids.append(tok.eos_id)

    if len(prompt_ids) > max_seq_len:
        prompt_ids    = prompt_ids[:max_seq_len - 1] + [tok.eos_id]
        was_truncated = True
    if len(resp_ids) > max_seq_len:
        resp_ids      = resp_ids[:max_seq_len - 1] + [tok.eos_id]
        was_truncated = True

    return TokenizedSample(
    prompt_ids,  resp_ids,
    len(prompt_ids), len(resp_ids)
    ) ,    was_truncated


# ── main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(record_gen, tok: HFTokenizerWrapper, out_dir: str,
                 max_seq_len: int, shard_size: int):

    use_sharding = shard_size > 0
    shard_idx    = 0
    writer       = ShardWriter(max_seq_len)
    flusher      = ShardFlusher(max_seq_len)

    def shard_dir_for(idx):
        if use_sharding:
            return os.path.join(out_dir, f"shard_{idx:02d}")
        return out_dir

    for rec in record_gen:
        sample, truncated = tokenize_record(rec, tok, max_seq_len)
        writer.add(sample, truncated)

        if use_sharding and writer.n_records() >= shard_size:
            snap = writer.snapshot_and_reset()
            flusher.submit(shard_dir_for(shard_idx), snap, shard_idx)
            shard_idx += 1

    # flush final partial shard
    if writer.n_records() > 0:
        snap = writer.snapshot_and_reset()
        flusher.submit(shard_dir_for(shard_idx), snap, shard_idx)

    flusher.join()

    # write manifest
    entries = sorted(flusher.manifest_entries(), key=lambda e: e["shard"])
    manifest = {
        "max_seq_len":  max_seq_len,
        "pad_token_id": tok.pad_id,
        "shards":       entries,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total = sum(e["n_samples"] for e in entries)
    print(f"[tokenize] manifest written → {manifest_path}")
    print(f"[tokenize] {total} samples in {len(entries)} shard(s) → {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Multi-source streaming tokenizer with sharded binary output"
    )
    parser.add_argument("--input", action="append", required=True,
                        help="HuggingFace dataset name or local .jsonl path. "
                             "Repeat for multiple sources.")
    parser.add_argument("--split",  default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer",  default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-seq-len",      type=int, default=512)
    parser.add_argument("--shard-size",       type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--shuffle-buffer-size", type=int, default=DEFAULT_SHUFFLE_BUF)
    parser.add_argument("--instruction-col", default=None)
    parser.add_argument("--response-col",    default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[tokenize] loading tokenizer '{args.tokenizer}' ...")
    tok = load_tokenizer(args.tokenizer)
    print(f"[tokenize] tokenizer loaded  eos={tok.eos_id}  pad={tok.pad_id}")

    if len(args.input) == 1:
        # ── single source — no threading needed ──────────────────────
        inp = args.input[0]
        if inp.endswith(".jsonl") or os.path.isfile(inp):
            print(f"[tokenize] single source: local JSONL → {inp}")
            gen = stream_jsonl(inp)
        else:
            print(f"[tokenize] single source: HuggingFace → {inp} (split={args.split})")
            gen = stream_hf_dataset(inp, args.split,
                                    args.instruction_col, args.response_col)
        run_pipeline(gen, tok, args.output_dir,
                     args.max_seq_len, args.shard_size)

    else:
        # ── multiple sources — SourceThread + ShuffleBuffer ──────────
        shared_queue = queue.Queue(maxsize=args.shuffle_buffer_size * 2)
        threads = []

        for inp in args.input:
            if inp.endswith(".jsonl") or os.path.isfile(inp):
                print(f"[tokenize] source: local JSONL → {inp}")
                gen = stream_jsonl(inp)
            else:
                print(f"[tokenize] source: HuggingFace → {inp} (split={args.split})")
                gen = stream_hf_dataset(inp, args.split,
                                        args.instruction_col, args.response_col)
            t = SourceThread(gen, shared_queue, name=inp)
            threads.append(t)

        print(f"[tokenize] starting {len(threads)} source thread(s) ...")
        for t in threads:
            t.start()

        shuffle_buf = ShuffleBuffer(shared_queue, n_sources=len(threads),
                                    size=args.shuffle_buffer_size)

        run_pipeline(shuffle_buf, tok, args.output_dir,
                     args.max_seq_len, args.shard_size)

        for t in threads:
            t.join()

    print("[tokenize] all done.")


if __name__ == "__main__":
    main()
