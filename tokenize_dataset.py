from __future__ import annotations

import argparse
import struct
from typing import List, Tuple, Optional
import os

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"


# ── Column name candidates ────────────────────────────────────────────────────
# Add any new dataset column patterns here and they will be picked up automatically

INSTRUCTION_CANDIDATES = [
    "prompt",
    "instruction",
    "input",
    "question",
    "human",
    "user",
    "context",
    "query",
]

RESPONSE_CANDIDATES = [
    "completion",
    "response",
    "output",
    "answer",
    "assistant",
    "gpt",
    "text",
    "target",
]


def detect_columns(row: dict) -> Optional[Tuple[str, str]]:
    """
    Try to detect instruction and response columns from a dataset row.
    Returns (instruction_col, response_col) or None if not found.
    """
    keys = [k.lower() for k in row.keys()]
    original_keys = list(row.keys())

    # build a map from lowercase key to original key
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
    """
    Handle the conversational format where data is stored as:
    messages: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    
    Also handles other common conversational column names:
    conversations, conversation, dialogue, chat
    """
    # possible column names for conversational format
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

            # normalize role name
            role = msg.get("role", msg.get("from", "")).lower().strip()
            content = msg.get("content", msg.get("value", "")).strip()

            if not content:
                continue

            # first human/user message is the instruction
            if role in ("user", "human", "instruction") and instruction is None:
                instruction = content

            # first assistant/gpt message is the response
            elif role in ("assistant", "gpt", "bot", "system_response") and response is None:
                response = content

        if instruction and response:
            return instruction, response

    return None


def extract_pair(row: dict, inst_col: Optional[str], resp_col: Optional[str]) -> Tuple[str, str]:
    """
    Extract instruction and response from a row using all available strategies.
    Priority:
      1. explicit column names provided by user via CLI
      2. conversational format (messages list)
      3. auto-detected flat columns
    """

    # strategy 1 — user explicitly provided column names
    if inst_col and resp_col:
        instruction = str(row.get(inst_col, "")).strip()
        response = str(row.get(resp_col, "")).strip()
        if instruction and response:
            return instruction, response

    # strategy 2 — conversational format
    result = extract_from_messages(row)
    if result:
        return result

    # strategy 3 — auto detect flat columns
    detected = detect_columns(row)
    if detected:
        i_col, r_col = detected
        instruction = str(row.get(i_col, "")).strip()
        response = str(row.get(r_col, "")).strip()
        if instruction and response:
            return instruction, response

    # strategy 4 — alpaca style with instruction + input + output
    if "instruction" in row and "output" in row:
        instruction = str(row.get("instruction", "")).strip()
        extra_input = str(row.get("input", "")).strip()
        response = str(row.get("output", "")).strip()
        # combine instruction and input if input exists
        if extra_input:
            instruction = f"{instruction}\n{extra_input}"
        if instruction and response:
            return instruction, response

    raise ValueError(
        f"Could not extract instruction/response pair from row.\n"
        f"Available columns: {list(row.keys())}\n"
        f"Row preview: { {k: str(v)[:100] for k, v in row.items()} }"
    )


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

    eos_id = tokenizer.eos_token_id

    print(f"[tokenize] Loading dataset: {args.dataset} / {args.split}")

    dataset = load_dataset(
        args.dataset,
        trust_remote_code=True
    )[args.split]

    print(f"[tokenize] {len(dataset):,} samples")

    # ── Auto-detect format from first row ────────────────────────────────────
    first_row = dataset[0]
    print(f"[tokenize] Detected columns: {list(first_row.keys())}")

    # check if user provided explicit column names
    inst_col = args.instruction_col if hasattr(args, "instruction_col") else None
    resp_col = args.response_col if hasattr(args, "response_col") else None

    # dry run on first row to validate extraction works before processing everything
    try:
        inst_preview, resp_preview = extract_pair(first_row, inst_col, resp_col)
        print(f"[tokenize] Instruction preview : {inst_preview[:80]}...")
        print(f"[tokenize] Response preview    : {resp_preview[:80]}...")
    except ValueError as e:
        print(f"[tokenize] ERROR: {e}")
        print("[tokenize] Use --instruction_col and --response_col to specify columns manually.")
        return

    instruction_sequences: List[List[int]] = []
    response_sequences: List[List[int]] = []

    max_len = args.max_seq_len
    skipped = 0

    for idx, row in enumerate(dataset):

        try:
            instruction, response = extract_pair(row, inst_col, resp_col)
        except ValueError:
            skipped += 1
            continue

        # skip empty pairs
        if not instruction.strip() or not response.strip():
            skipped += 1
            continue

        inst_ids = tokenizer.encode(
            instruction,
            add_special_tokens=True,
            truncation=True,
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
            raise RuntimeError(
                f"instruction length exceeds uint16 limit: {len(inst_ids)}"
            )

        if len(resp_ids) > 65535:
            raise RuntimeError(
                f"response length exceeds uint16 limit: {len(resp_ids)}"
            )

        instruction_sequences.append(inst_ids)
        response_sequences.append(resp_ids)

        if (idx + 1) % 1000 == 0:
            print(
                f"tokenized {idx + 1:,}/{len(dataset):,}",
                end="\r",
                flush=True,
            )

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
    response_tokens = sum(len(seq) for seq in response_sequences)

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

    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer name")
    parser.add_argument("--instructions_output", default="instructions.bin")
    parser.add_argument("--responses_output", default="responses.bin")
    parser.add_argument("--max_seq_len", type=int, default=512)

    # optional explicit column override — used when auto detection fails
    parser.add_argument(
        "--instruction_col",
        default=None,
        help="Explicit column name for instructions (optional, auto-detected if not set)"
    )
    parser.add_argument(
        "--response_col",
        default=None,
        help="Explicit column name for responses (optional, auto-detected if not set)"
    )

    args = parser.parse_args()
    tokenize(args)


if __name__ == "__main__":
    main()