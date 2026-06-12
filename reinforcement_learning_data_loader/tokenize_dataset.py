#!/usr/bin/env python3

import argparse
import json
import os
import struct
import sys


MAX_SEQ_LEN_HARD_CAP = 65535  # uint16 length field
DEFAULT_TOKENIZER = "inceptionai/jais-family-590m"

def load_tokenizer(name_or_path):
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer '{name_or_path}': {e}\n"
            f"Make sure `transformers` is installed (pip install transformers) "
            f"and that you have network access to download the tokenizer files."
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


def write_binary(path, sequences, max_len=None):
    """sequences: list[list[int]]"""
    n_written = 0
    n_truncated = 0

    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(sequences)))
        for seq in sequences:
            if max_len is not None and len(seq) > max_len:
                seq = seq[:max_len]
                n_truncated += 1
            if len(seq) > MAX_SEQ_LEN_HARD_CAP:
                seq = seq[:MAX_SEQ_LEN_HARD_CAP]
                n_truncated += 1

            f.write(struct.pack("<H", len(seq)))
            for tok in seq:
                f.write(struct.pack("<I", tok))
            n_written += 1

    print(f"[tokenize_dataset] wrote {n_written} sequences to {path}"
          + (f" ({n_truncated} truncated)" if n_truncated else ""))


def run_sft(records, tok, out_dir, max_len):
    prompts, responses = [], []
    for r in records:
        prompts.append(tok.encode(r["prompt"], add_eos=False))
        responses.append(tok.encode(r["response"], add_eos=True))

    write_binary(os.path.join(out_dir, "prompts.bin"), prompts, max_len)
    write_binary(os.path.join(out_dir, "responses.bin"), responses, max_len)


def run_preference(records, tok, out_dir, max_len, label):
    prompts, chosen, rejected = [], [], []
    for r in records:
        prompts.append(tok.encode(r["prompt"], add_eos=False))
        chosen.append(tok.encode(r["chosen"], add_eos=True))
        rejected.append(tok.encode(r["rejected"], add_eos=True))

    write_binary(os.path.join(out_dir, "prompts.bin"), prompts, max_len)
    write_binary(os.path.join(out_dir, "chosen.bin"), chosen, max_len)
    write_binary(os.path.join(out_dir, "rejected.bin"), rejected, max_len)
    print(f"[tokenize_dataset] {label} mode: prompts.bin / chosen.bin / "
          f"rejected.bin written to {out_dir}")


def run_rollout(records, tok, out_dir, max_prompt_len):
    prompts = []
    for r in records:
        prompts.append(tok.encode(r["prompt"], add_eos=False))

    write_binary(os.path.join(out_dir, "prompts.bin"), prompts, max_prompt_len)
    print(f"[tokenize_dataset] rollout mode: prompts.bin written to {out_dir} "
          f"(prompt-only, right-truncated to {max_prompt_len})")

def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output-dir", required=True,    help="Directory to write .bin files into")
      parser.add_argument("--mode", required=True,        choices=["sft", "reward_model", "dpo", "rollout"],   help="Which binary files to produce")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,  help=f"HuggingFace tokenizer name/path "     f"(default: {DEFAULT_TOKENIZER})")
    parser.add_argument("--max-seq-len", type=int, default=2048,   help="Max length for sft/reward_model/dpo sequences "    "(applied independently per field)")
    parser.add_argument("--max-prompt-len", type=int, default=512, help="Max prompt length for rollout mode")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"[tokenize_dataset] loaded {len(records)} records from {args.input}")

    print(f"[tokenize_dataset] loading tokenizer '{args.tokenizer}' ...")
    tok = load_tokenizer(args.tokenizer)
    print(f"[tokenize_dataset] tokenizer loaded (eos_token_id={tok.eos_token_id})")

    if args.mode == "sft":
        run_sft(records, tok, args.output_dir, args.max_seq_len)
    elif args.mode in ("reward_model", "dpo"):
        run_preference(records, tok, args.output_dir, args.max_seq_len, args.mode)
    elif args.mode == "rollout":
        run_rollout(records, tok, args.output_dir, args.max_prompt_len)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()























"""
tokenize_dataset.py — Unified tokenization script for the
post-training data loader (SFT / REWARD_MODEL / DPO / ROLLOUT).

Binary file format (identical for every file produced):

    uint32  n_samples
    [per sample]
        uint16  length
        uint32  token_ids[length]

Modes and the files they produce:

    sft           -> prompts.bin, responses.bin
    reward_model  -> prompts.bin, chosen.bin, rejected.bin
    dpo           -> prompts.bin, chosen.bin, rejected.bin   (same as reward_model)
    rollout       -> prompts.bin

Input data format (JSONL), one JSON object per line:

    sft:
        {"prompt": "...", "response": "..."}

    reward_model / dpo:
        {"prompt": "...", "chosen": "...", "rejected": "..."}

    rollout:
        {"prompt": "..."}

Tokenizer:
    Defaults to the real HuggingFace tokenizer for
    "inceptionai/jais-family-590m" (loaded via AutoTokenizer with
    trust_remote_code=True, since Jais ships a custom tokenizer class).

    Override with --tokenizer <name_or_path> to use any other
    HuggingFace tokenizer.

Install requirements:
    pip install transformers
"""
