# instructions.bin

# [uint32 n_samples]

 # [uint16 len]
 # [token ids ...]

 # [uint16 len]
 # [token ids ...]

# responses.bin

# [uint32 n_samples]

# [uint16 len]
 # [token ids ...]

 # [uint16 len]
 # [token ids ...]


from __future__ import annotations

import argparse
import struct
from typing import List


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

    dataset = load_dataset(args.dataset)[args.split]

    print(f"[tokenize] {len(dataset):,} samples")

    instruction_sequences: List[List[int]] = []
    response_sequences: List[List[int]] = []

    max_len = args.max_seq_len

    for idx, row in enumerate(dataset):

        instruction = row["prompt"]
        response = row["completion"]

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

    print(f"[tokenize] Writing {args.instructions_output}")

    with open(args.instructions_output, "wb") as f:

        f.write(struct.pack("<I", len(instruction_sequences)))

        for seq in instruction_sequences:

            f.write(struct.pack("<H", len(seq)))

            f.write(
                struct.pack(
                    f"<{len(seq)}I",
                    *seq,
                )
            )

    print(f"[tokenize] Writing {args.responses_output}")

    with open(args.responses_output, "wb") as f:

        f.write(struct.pack("<I", len(response_sequences)))

        for seq in response_sequences:

            f.write(struct.pack("<H", len(seq)))

            f.write(
                struct.pack(
                    f"<{len(seq)}I",
                    *seq,
                )
            )

    instruction_tokens = sum(
        len(seq)
        for seq in instruction_sequences
    )

    response_tokens = sum(
        len(seq)
        for seq in response_sequences
    )

    print()
    print("[tokenize] Done")
    print(f"Samples             : {len(dataset):,}")
    print(f"Instruction tokens  : {instruction_tokens:,}")
    print(f"Response tokens     : {response_tokens:,}")
    print(f"Instructions file   : {args.instructions_output}")
    print(f"Responses file      : {args.responses_output}")
    print(tokenizer.special_tokens_map)
    print(tokenizer.eos_token_id, tokenizer.pad_token_id)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--split",
        default="test",
    )

    parser.add_argument(
        "--tokenizer",
        default="gpt2",
    )

    parser.add_argument(
        "--instructions_output",
        default="instructions.bin",
    )

    parser.add_argument(
        "--responses_output",
        default="responses.bin",
    )

    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
    )

    args = parser.parse_args()

    tokenize(args)
    

if __name__ == "__main__":
    main()