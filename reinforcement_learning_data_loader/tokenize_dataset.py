#!/usr/bin/env python3
"""
tokenize_dataset.py — Streaming tokenization + sharding for the
post-training data loader (SFT / REWARD_MODEL / DPO / ROLLOUT).

CHANGES FROM ORIGINAL:
    - No longer loads the full JSONL into RAM before tokenizing.
      Records are consumed one at a time from a streaming generator.
    - Output is split into numbered shards (shard_00/, shard_01/, ...)
      each containing at most --shard-size records. This keeps peak
      RAM proportional to one shard, not the full dataset.
    - Input can be either:
        a) A HuggingFace dataset name  (e.g. "Anthropic/hh-rlhf")
           streamed via prepare_dataset.stream_dataset()
        b) A local JSONL file path     (e.g. "data/my_data.jsonl")
           streamed via prepare_dataset.stream_jsonl()
      The --input flag accepts both. If the value looks like a file
      path (ends with .jsonl or the path exists on disk) it is treated
      as a local JSONL file; otherwise it is treated as a HuggingFace
      dataset name.
    - The old --input / --output-dir / --mode / --tokenizer /
      --max-seq-len / --max-prompt-len flags are all preserved so
      existing notebook cells that call this script continue to work.

Binary shard format (identical to the original single-file format):

    uint32  n_samples          <- number of sequences in THIS shard
    [per sample]
        uint16  length
        uint32  token_ids[length]

Output directory layout:

    <output-dir>/
        shard_00/
            prompts.bin
            chosen.bin          (reward_model / dpo modes)
            rejected.bin        (reward_model / dpo modes)
            responses.bin       (sft mode)
        shard_01/
            ...
        manifest.json           <- lists all shards + sample counts

The manifest lets the C++ multi-dataset loader discover shards without
scanning the directory tree.

Backward compatibility:
    If --shard-size is set to 0 (or omitted and the dataset fits in
    one shard) the output layout is identical to the original:

        <output-dir>/
            prompts.bin
            chosen.bin
            rejected.bin

    so existing C++ code that reads a single flat directory still works.
"""

import argparse
import json
import os
import struct
import sys

MAX_SEQ_LEN_HARD_CAP = 65535   # uint16 length field
DEFAULT_TOKENIZER    = "inceptionai/jais-family-590m"
DEFAULT_SHARD_SIZE   = 10_000  # records per shard (0 = no sharding)


def load_tokenizer(name_or_path):
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer '{name_or_path}': {e}\n"
            "Make sure `transformers` is installed and you have network access."
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


# Binary shard writer
# Accumulates encoded sequences for one shard then flushes to disk.
# Peak RAM = one shard of token ID lists (typically ~20 MB).

class ShardWriter:
    """
    Collects encoded sequences for a single shard and writes them
    to disk when flush() is called.

    Usage:
        writer = ShardWriter(shard_dir, max_len)
        writer.add("prompts",   prompt_ids)
        writer.add("chosen",    chosen_ids)
        writer.add("rejected",  rejected_ids)
        ...
        writer.flush()   # writes shard_dir/prompts.bin etc. and resets
    """

    def __init__(self, out_dir, max_len):
        self.out_dir  = out_dir
        self.max_len  = max_len
        self._buffers = {}          # field_name -> list[list[int]]
        self._count   = 0
        self._n_truncated = 0

    def add(self, field, token_ids):
        #Add one encoded sequence to the named field buffer
        if self.max_len and len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
            self._n_truncated += 1
        if len(token_ids) > MAX_SEQ_LEN_HARD_CAP:
            token_ids = token_ids[:MAX_SEQ_LEN_HARD_CAP]
            self._n_truncated += 1
        self._buffers.setdefault(field, []).append(token_ids)

    def commit_record(self):
        #Call once per input record after all add() calls for that record.
        self._count += 1

    def n_records(self):
        return self._count

    def flush(self):
        #Write all buffered sequences to disk and reset the buffer.
        if self._count == 0:
            return {}

        os.makedirs(self.out_dir, exist_ok=True)
        written_files = {}

        for field, seqs in self._buffers.items():
            path = os.path.join(self.out_dir, f"{field}.bin")
            with open(path, "wb") as f:
                f.write(struct.pack("<I", len(seqs)))
                for seq in seqs:
                    f.write(struct.pack("<H", len(seq)))
                    if seq:
                        f.write(struct.pack(f"<{len(seq)}I", *seq))
            written_files[field] = path
            print(f"  [shard] wrote {len(seqs)} sequences → {path}"
                  + (f" ({self._n_truncated} truncated)" if self._n_truncated else ""))

        n = self._count
        # reset
        self._buffers      = {}
        self._count        = 0
        self._n_truncated  = 0
        return written_files, n


# Streaming tokenization modes
# Each run_* function accepts a record GENERATOR (not a list) so it
# never materialises more than one shard's records in RAM.

def _make_shard_dir(base_out_dir, shard_idx, use_sharding):
    """Return the directory to write a shard into."""
    if use_sharding:
        return os.path.join(base_out_dir, f"shard_{shard_idx:02d}")
    return base_out_dir   # flat layout — backward compatible


def run_sft(record_gen, tok, out_dir, max_len, shard_size):
    use_sharding = shard_size > 0
    manifest     = []
    shard_idx    = 0
    writer       = ShardWriter(_make_shard_dir(out_dir, shard_idx, use_sharding), max_len)

    for rec in record_gen:
        writer.add("prompts",   tok.encode(rec["prompt"],   add_eos=False))
        writer.add("responses", tok.encode(rec["response"], add_eos=True))
        writer.commit_record()

        if use_sharding and writer.n_records() >= shard_size:
            _, n = writer.flush()
            manifest.append({"shard": shard_idx, "n_samples": n,
                              "dir": _make_shard_dir(out_dir, shard_idx, use_sharding)})
            shard_idx += 1
            writer = ShardWriter(_make_shard_dir(out_dir, shard_idx, use_sharding), max_len)

    # flush final (possibly partial) shard
    if writer.n_records() > 0:
        _, n = writer.flush()
        manifest.append({"shard": shard_idx, "n_samples": n,
                          "dir": _make_shard_dir(out_dir, shard_idx, use_sharding)})

    _write_manifest(out_dir, "sft", manifest, use_sharding)
    print(f"[tokenize_dataset] sft mode: {sum(e['n_samples'] for e in manifest)} "
          f"records written across {len(manifest)} shard(s) in {out_dir}")


def run_preference(record_gen, tok, out_dir, max_len, shard_size, label):
    use_sharding = shard_size > 0
    manifest     = []
    shard_idx    = 0
    writer       = ShardWriter(_make_shard_dir(out_dir, shard_idx, use_sharding), max_len)

    for rec in record_gen:
        writer.add("prompts",  tok.encode(rec["prompt"],   add_eos=False))
        writer.add("chosen",   tok.encode(rec["chosen"],   add_eos=True))
        writer.add("rejected", tok.encode(rec["rejected"], add_eos=True))
        writer.commit_record()

        if use_sharding and writer.n_records() >= shard_size:
            _, n = writer.flush()
            manifest.append({"shard": shard_idx, "n_samples": n,
                              "dir": _make_shard_dir(out_dir, shard_idx, use_sharding)})
            shard_idx += 1
            writer = ShardWriter(_make_shard_dir(out_dir, shard_idx, use_sharding), max_len)

    if writer.n_records() > 0:
        _, n = writer.flush()
        manifest.append({"shard": shard_idx, "n_samples": n,
                          "dir": _make_shard_dir(out_dir, shard_idx, use_sharding)})

    _write_manifest(out_dir, label, manifest, use_sharding)
    total = sum(e["n_samples"] for e in manifest)
    print(f"[tokenize_dataset] {label} mode: {total} records written across "
          f"{len(manifest)} shard(s) → prompts / chosen / rejected in {out_dir}")


def run_rollout(record_gen, tok, out_dir, max_prompt_len, shard_size):
    use_sharding = shard_size > 0
    manifest     = []
    shard_idx    = 0
    writer       = ShardWriter(_make_shard_dir(out_dir, shard_idx, use_sharding),
                               max_prompt_len)

    for rec in record_gen:
        writer.add("prompts", tok.encode(rec["prompt"], add_eos=False))
        writer.commit_record()

        if use_sharding and writer.n_records() >= shard_size:
            _, n = writer.flush()
            manifest.append({"shard": shard_idx, "n_samples": n,
                              "dir": _make_shard_dir(out_dir, shard_idx, use_sharding)})
            shard_idx += 1
            writer = ShardWriter(_make_shard_dir(out_dir, shard_idx, use_sharding),
                                 max_prompt_len)

    if writer.n_records() > 0:
        _, n = writer.flush()
        manifest.append({"shard": shard_idx, "n_samples": n,
                          "dir": _make_shard_dir(out_dir, shard_idx, use_sharding)})

    _write_manifest(out_dir, "rollout", manifest, use_sharding)
    total = sum(e["n_samples"] for e in manifest)
    print(f"[tokenize_dataset] rollout mode: {total} prompts written across "
          f"{len(manifest)} shard(s) in {out_dir}")


# Manifest
def _write_manifest(out_dir, mode, manifest, use_sharding):
    """Write manifest.json so the C++ loader can discover shards."""
    if not use_sharding:
        return   # flat layout — no manifest needed, C++ reads directly
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"mode": mode, "shards": manifest}, f, indent=2)
    print(f"[tokenize_dataset] manifest written → {path}")

# Input source resolution
# Decides whether --input is a HuggingFace dataset name or a local file.
def _resolve_input(input_arg, split):
    """
    Returns a generator of normalized {prompt, chosen, rejected} records.

    Rules:
        - If input_arg ends with .jsonl OR the path exists on disk
          → treat as local JSONL file via stream_jsonl()
        - Otherwise
          → treat as HuggingFace dataset name via stream_dataset()
    """
    import prepare_dataset as pd

    is_local = input_arg.endswith(".jsonl") or os.path.isfile(input_arg)

    if is_local:
        print(f"[tokenize_dataset] input: local JSONL → {input_arg}")
        return pd.stream_jsonl(input_arg)
    else:
        print(f"[tokenize_dataset] input: HuggingFace dataset → {input_arg} "
              f"(split={split})")
        return pd.stream_dataset(input_arg, split=split)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter  )
    parser.add_argument(
        "--input", required=True,
    help="HuggingFace dataset name (e.g. 'Anthropic/hh-rlhf') "
             "OR path to a local .jsonl file" )
    parser.add_argument(
               "--split", default="train", help="HuggingFace split to use (ignored for local files, default: train)" )
    parser.add_argument(  "--output-dir", required=True,
     help="Root directory to write shard subdirectories into")
    parser.add_argument(
        "--mode", required=True,
  choices=["sft", "reward_model", "dpo", "rollout"], help="Which binary files to produce"  )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,help=f"HuggingFace tokenizer name/path (default: {DEFAULT_TOKENIZER})" )
    parser.add_argument(  "--max-seq-len", type=int, default=2048, help="Max token length for sft/reward_model/dpo sequences")
    parser.add_argument( "--max-prompt-len", type=int, default=512, help="Max prompt token length for rollout mode" )
    parser.add_argument(
     "--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
    help="Records per shard directory (0 = no sharding, flat layout "
             "identical to the original script)" )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[tokenize_dataset] loading tokenizer '{args.tokenizer}' ...")
    tok = load_tokenizer(args.tokenizer)
    print(f"[tokenize_dataset] tokenizer loaded (eos_token_id={tok.eos_token_id})")
    record_gen = _resolve_input(args.input, args.split)
    if args.mode == "sft":
        run_sft(record_gen, tok, args.output_dir, args.max_seq_len, args.shard_size)

    elif args.mode in ("reward_model", "dpo"):
        run_preference(record_gen, tok, args.output_dir,
                       args.max_seq_len, args.shard_size, args.mode)
    elif args.mode == "rollout":
        run_rollout(record_gen, tok, args.output_dir,
                    args.max_prompt_len, args.shard_size)
if __name__ == "__main__":
    main()