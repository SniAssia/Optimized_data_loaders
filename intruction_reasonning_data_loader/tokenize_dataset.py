from __future__ import annotations

import argparse
import struct
from typing import List, Tuple, Optional
import os

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

INSTRUCTION_CANDIDATES = [
    "prompt", "instruction", "input", "question",
    "human", "user", "context", "query",
]

RESPONSE_CANDIDATES = [
    "completion", "response", "output", "answer",
    "assistant", "gpt", "text", "target",
]


def detect_columns(row: dict) -> Optional[Tuple[str, str]]:
    keys = [k.lower() for k in row.keys()]
    original_keys = list(row.keys())
    key_map = {k.lower(): k for k in original_keys}
    inst_col = None
    resp_col = None
    for candidate in INSTRUCTION_CANDIDATES:
        if candidate in keys:
            inst_col = key_map[candidate]
            break
    for candidate in RESPONSE_CANDIDATES:
        if candidate in keys:
            resp_col = key_map[candidate]
            break
    if inst_col and resp_col:
        return inst_col, resp_col
    return None


def extract_from_messages(row: dict) -> Optional[Tuple[str, str]]:
    conv_candidates = ["messages", "conversations", "conversation", "dialogue", "chat"]
    for col in conv_candidates:
        if col not in row:
            continue
        messages = row[col]
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        instruction = None
        response = None
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", msg.get("from", "")).lower().strip()
            content = msg.get("content", msg.get("value", "")).strip()
            if not content:
                continue
            if role in ("user", "human", "instruction") and instruction is None:
                instruction = content
            elif role in ("assistant", "gpt", "bot", "system_response") and response is None:
                response = content
        if instruction and response:
            return instruction, response
    return None


def extract_pair(row: dict, inst_col: Optional[str], resp_col: Optional[str]) -> Tuple[str, str]:
    if inst_col and resp_col:
        instruction = str(row.get(inst_col, "")).strip()
        response = str(row.get(resp_col, "")).strip()
        if instruction and response:
            return instruction, response

    result = extract_from_messages(row)
    if result:
        return result

    if "instruction" in row and "output" in row:
        instruction = str(row.get("instruction", "")).strip()
        extra_input = str(row.get("input", "")).strip()
        response = str(row.get("output", "")).strip()
        if extra_input:
            instruction = f"{instruction}\n{extra_input}"
        if instruction and response:
            return instruction, response

    detected = detect_columns(row)
    if detected:
        i_col, r_col = detected
        instruction = str(row.get(i_col, "")).strip()
        response = str(row.get(r_col, "")).strip()
        if instruction and response:
            return instruction, response

    raise ValueError(
        f"Could not extract instruction/response pair from row.\n"
        f"Available columns: {list(row.keys())}\n"
        f"Row preview: { {k: str(v)[:100] for k, v in row.items()} }"
    )


def compute_length_stats(dataset, tokenizer, inst_col, resp_col, max_len):
    """
    Compute length statistics BEFORE truncation.
    Reports how many sequences exceed max_len and by how much.
    """
    print(f"\n[stats] Analyzing sequence lengths before truncation...")

    inst_lengths = []
    resp_lengths = []
    combined_lengths = []

    inst_exceeded     = 0
    resp_exceeded     = 0
    combined_exceeded = 0

    inst_exceeded_total_tokens     = 0
    resp_exceeded_total_tokens     = 0
    combined_exceeded_total_tokens = 0

    for idx, row in enumerate(dataset):

        try:
            instruction, response = extract_pair(row, inst_col, resp_col)
        except ValueError:
            continue

        if not instruction.strip() or not response.strip():
            continue

        # encode WITHOUT truncation to get true lengths
        inst_ids = tokenizer.encode(instruction, add_special_tokens=True)
        resp_ids = tokenizer.encode(response,    add_special_tokens=True)

        inst_len     = len(inst_ids)
        resp_len     = len(resp_ids)
        combined_len = inst_len + resp_len

        inst_lengths.append(inst_len)
        resp_lengths.append(resp_len)
        combined_lengths.append(combined_len)

        # count instruction sequences exceeding max_len
        if inst_len > max_len:
            inst_exceeded += 1
            inst_exceeded_total_tokens += (inst_len - max_len)

        # count response sequences exceeding max_len
        if resp_len > max_len:
            resp_exceeded += 1
            resp_exceeded_total_tokens += (resp_len - max_len)

        # count combined sequences exceeding max_len
        if combined_len > max_len:
            combined_exceeded += 1
            combined_exceeded_total_tokens += (combined_len - max_len)

        if (idx + 1) % 5000 == 0:
            print(f"  analyzed {idx + 1:,}/{len(dataset):,}", end="\r", flush=True)

    print()

    total = len(inst_lengths)
    if total == 0:
        print("[stats] No valid samples found.")
        return

    # ── helper to compute percentiles ────────────────────────────────────────
    def percentile(sorted_list, p):
        idx = int(len(sorted_list) * p / 100)
        idx = min(idx, len(sorted_list) - 1)
        return sorted_list[idx]

    inst_sorted     = sorted(inst_lengths)
    resp_sorted     = sorted(resp_lengths)
    combined_sorted = sorted(combined_lengths)

    print()
    print("=" * 60)
    print("  SEQUENCE LENGTH STATISTICS  (before truncation)")
    print("=" * 60)
    print(f"  Total samples analyzed : {total:,}")
    print(f"  max_seq_len threshold  : {max_len}")
    print()

    print(f"  {'Metric':<20} {'Instruction':>14} {'Response':>14} {'Combined':>14}")
    print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*14}")
    print(f"  {'Min':<20} {min(inst_lengths):>14,} {min(resp_lengths):>14,} {min(combined_lengths):>14,}")
    print(f"  {'Max':<20} {max(inst_lengths):>14,} {max(resp_lengths):>14,} {max(combined_lengths):>14,}")
    print(f"  {'Mean':<20} {sum(inst_lengths)/total:>14.1f} {sum(resp_lengths)/total:>14.1f} {sum(combined_lengths)/total:>14.1f}")
    print(f"  {'Median (p50)':<20} {percentile(inst_sorted,50):>14,} {percentile(resp_sorted,50):>14,} {percentile(combined_sorted,50):>14,}")
    print(f"  {'p75':<20} {percentile(inst_sorted,75):>14,} {percentile(resp_sorted,75):>14,} {percentile(combined_sorted,75):>14,}")
    print(f"  {'p90':<20} {percentile(inst_sorted,90):>14,} {percentile(resp_sorted,90):>14,} {percentile(combined_sorted,90):>14,}")
    print(f"  {'p95':<20} {percentile(inst_sorted,95):>14,} {percentile(resp_sorted,95):>14,} {percentile(combined_sorted,95):>14,}")
    print(f"  {'p99':<20} {percentile(inst_sorted,99):>14,} {percentile(resp_sorted,99):>14,} {percentile(combined_sorted,99):>14,}")
    print()

    print(f"  TRUNCATION IMPACT (sequences exceeding max_seq_len={max_len})")
    print(f"  {'-'*58}")
    print(f"  {'':20} {'Instruction':>14} {'Response':>14} {'Combined':>14}")
    print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*14}")
    print(f"  {'Count exceeded':<20} {inst_exceeded:>14,} {resp_exceeded:>14,} {combined_exceeded:>14,}")
    print(f"  {'% of total':<20} {100*inst_exceeded/total:>13.1f}% {100*resp_exceeded/total:>13.1f}% {100*combined_exceeded/total:>13.1f}%")
    print(f"  {'Tokens lost':<20} {inst_exceeded_total_tokens:>14,} {resp_exceeded_total_tokens:>14,} {combined_exceeded_total_tokens:>14,}")
    print()

    # ── bucket distribution preview ───────────────────────────────────────────
    boundaries = [64, 128, 256, 512, 1024, 2048]
    print(f"  COMBINED LENGTH DISTRIBUTION across buckets")
    print(f"  {'-'*40}")
    prev = 0
    for b in boundaries:
        count = sum(1 for l in combined_lengths if prev < l <= b)
        pct   = 100 * count / total
        bar   = "█" * int(pct / 2)
        print(f"  ({prev:>4}, {b:>4}] : {count:>6,} samples  ({pct:5.1f}%)  {bar}")
        prev = b
    over = sum(1 for l in combined_lengths if l > boundaries[-1])
    pct  = 100 * over / total
    bar  = "█" * int(pct / 2)
    print(f"  (>{boundaries[-1]:>4}      : {over:>6,} samples  ({pct:5.1f}%)  {bar}")
    print("=" * 60)
    print()


def tokenize(args) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"[tokenize] Loading tokenizer: {args.tokenizer}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    eos_id  = tokenizer.eos_token_id
    max_len = args.max_seq_len

    print(f"[tokenize] Loading dataset: {args.dataset} / {args.split}")

    dataset = load_dataset(
        args.dataset,
        trust_remote_code=True
    )[args.split]

    print(f"[tokenize] {len(dataset):,} samples")

    first_row = dataset[0]
    print(f"[tokenize] Detected columns: {list(first_row.keys())}")

    inst_col = args.instruction_col if hasattr(args, "instruction_col") else None
    resp_col = args.response_col    if hasattr(args, "response_col")    else None

    try:
        inst_preview, resp_preview = extract_pair(first_row, inst_col, resp_col)
        print(f"[tokenize] Instruction preview : {inst_preview[:80]}...")
        print(f"[tokenize] Response preview    : {resp_preview[:80]}...")
    except ValueError as e:
        print(f"[tokenize] ERROR: {e}")
        print("[tokenize] Use --instruction_col and --response_col to specify columns manually.")
        return

    # ── run length statistics BEFORE tokenizing ───────────────────────────────
    if not args.skip_stats:
        compute_length_stats(dataset, tokenizer, inst_col, resp_col, max_len)

    instruction_sequences: List[List[int]] = []
    response_sequences:    List[List[int]] = []
    skipped = 0

    for idx, row in enumerate(dataset):

        try:
            instruction, response = extract_pair(row, inst_col, resp_col)
        except ValueError:
            skipped += 1
            continue

        if not instruction.strip() or not response.strip():
            skipped += 1
            continue

        inst_ids = tokenizer.encode(
            instruction,
            add_special_tokens=True,
            truncation=False,
            max_length=max_len,
        )

        resp_ids = tokenizer.encode(
            response,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
        )

        if not inst_ids:
            inst_ids = [eos_id]
        if not resp_ids:
            resp_ids = [eos_id]

        if inst_ids[-1] != eos_id:
            inst_ids = inst_ids[: max_len - 1] + [eos_id]
        if resp_ids[-1] != eos_id:
            resp_ids = resp_ids[: max_len - 1] + [eos_id]

        if len(inst_ids) > 65535:
            raise RuntimeError(f"instruction length exceeds uint16 limit: {len(inst_ids)}")
        if len(resp_ids) > 65535:
            raise RuntimeError(f"response length exceeds uint16 limit: {len(resp_ids)}")

        instruction_sequences.append(inst_ids)
        response_sequences.append(resp_ids)

        if (idx + 1) % 1000 == 0:
            print(f"tokenized {idx + 1:,}/{len(dataset):,}", end="\r", flush=True)

    print()

    if skipped > 0:
        print(f"[tokenize] Skipped {skipped:,} samples with missing/empty fields")

    print(f"[tokenize] Writing {args.instructions_output}")
    with open(args.instructions_output, "wb") as f:
        f.write(struct.pack("<I", len(instruction_sequences)))
        for seq in instruction_sequences:
            f.write(struct.pack("<H", len(seq)))
            f.write(struct.pack(f"<{len(seq)}I", *seq))

    print(f"[tokenize] Writing {args.responses_output}")
    with open(args.responses_output, "wb") as f:
        f.write(struct.pack("<I", len(response_sequences)))
        for seq in response_sequences:
            f.write(struct.pack("<H", len(seq)))
            f.write(struct.pack(f"<{len(seq)}I", *seq))

    instruction_tokens = sum(len(seq) for seq in instruction_sequences)
    response_tokens    = sum(len(seq) for seq in response_sequences)

    print()
    print("[tokenize] Done")
    print(f"Samples processed   : {len(instruction_sequences):,}")
    print(f"Samples skipped     : {skipped:,}")
    print(f"Instruction tokens  : {instruction_tokens:,}")
    print(f"Response tokens     : {response_tokens:,}")
    print(f"Instructions file   : {args.instructions_output}")
    print(f"Responses file      : {args.responses_output}")
    print(f"Tokenizer EOS id    : {tokenizer.eos_token_id}")
    print(f"Tokenizer PAD id    : {tokenizer.pad_token_id}")


def main():

    parser = argparse.ArgumentParser(
        description="Tokenize instruction-response datasets into binary format for C++ dataloader"
    )

    parser.add_argument("--dataset",              required=True)
    parser.add_argument("--split",                default="train")
    parser.add_argument("--tokenizer",            default="inceptionai/jais-family-590m")
    parser.add_argument("--instructions_output",  default="instructions.bin")
    parser.add_argument("--responses_output",     default="responses.bin")
    parser.add_argument("--max_seq_len",          type=int, default=512)
    parser.add_argument("--instruction_col",      default=None)
    parser.add_argument("--response_col",         default=None)
    parser.add_argument(
        "--skip_stats",
        action="store_true",
        default=False,
        help="Skip the length statistics pass (faster, use when stats are not needed)"
    )

    args = parser.parse_args()
    tokenize(args)


if __name__ == "__main__":
    main()