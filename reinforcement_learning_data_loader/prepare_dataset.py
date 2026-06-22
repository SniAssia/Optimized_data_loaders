#!/usr/bin/env python3
import argparse
# Per-dataset adapters
# Each adapter receives one raw HuggingFace example dict and returns either
# a normalized dict  {"prompt": str, "chosen": str, "rejected": str}
# or None to signal that this example should be skipped.
def adapt_hh_rlhf(example):
    def split_last_assistant(text):
        idx = text.rfind("\n\nAssistant:")
        if idx == -1:
            return None, None
        prompt   = text[:idx].strip()
        response = text[idx + len("\n\nAssistant:"):].strip()
        return prompt, response

    prompt, chosen_resp=split_last_assistant(example["chosen"])
    _, rejected_resp=split_last_assistant(example["rejected"])

    if not prompt or not chosen_resp or not rejected_resp:
        return None

    return {"prompt": prompt, "chosen": chosen_resp, "rejected": rejected_resp}


def adapt_ultrafeedback(example):
    chosen= example["chosen"]
    rejected= example["rejected"]

    chosen_text   = chosen[-1]["content"]   if isinstance(chosen,   list) else chosen
    rejected_text = rejected[-1]["content"] if isinstance(rejected, list) else rejected

    if not example.get("prompt") or not chosen_text or not rejected_text:
        return None

    return {"prompt": example["prompt"], "chosen": chosen_text, "rejected": rejected_text}
def adapt_orca_dpo(example):
    prompt = example.get("system", "") + "\n" + example.get("question", "")
    chosen   = example.get("chosen", "")
    rejected = example.get("rejected", "")
    if not prompt.strip() or not chosen or not rejected:
        return None
    return {"prompt": prompt.strip(), "chosen": chosen, "rejected": rejected}
def adapt_ultrainteract(example):
    trajectory = example.get("trajectory", [])
    chosen   = example.get("chosen", "")
    rejected = example.get("rejected", "")

    if not trajectory or not chosen or not rejected:
        return None

    # trajectory is a list of {from, value} turns
    # concatenate all turns into a single prompt string
    prompt_parts = []
    for turn in trajectory:
        role  = turn.get("from", "")
        value = turn.get("value", "").strip()
        if role == "user":
            prompt_parts.append(f"User: {value}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {value}")

    prompt = "\n".join(prompt_parts).strip()

    if not prompt or not chosen.strip() or not rejected.strip():
        return None

    return {"prompt": prompt, "chosen": chosen.strip(), "rejected": rejected.strip()}

def adapt_shp(example):
    if example["labels"] == 1:
        chosen, rejected = example["human_ref_A"], example["human_ref_B"]
    else:
        chosen, rejected = example["human_ref_B"], example["human_ref_A"]

    if not example.get("history") or not chosen or not rejected:
        return None

    return {"prompt": example["history"], "chosen": chosen, "rejected": rejected}


# Registry: HuggingFace dataset name -> adapter function
ADAPTERS = {
    "Anthropic/hh-rlhf":                    adapt_hh_rlhf,
    "HuggingFaceH4/ultrafeedback_binarized": adapt_ultrafeedback,
    "stanfordnlp/SHP":                       adapt_shp,
     "openbmb/UltraInteract_pair":            adapt_ultrainteract, 
     "Intel/orca_dpo_pairs" : adapt_orca_dpo
}

# Public streaming API
def stream_dataset(dataset_name, split="train"):
    """
    Generator — yields one normalized record dict at a time.

    {"prompt": str, "chosen": str, "rejected": str}

    Records that fail adapter validation are silently skipped.
    The HuggingFace dataset is loaded in streaming mode so only
    one page of raw examples is held in RAM at any moment.

    Args:
        dataset_name: key in ADAPTERS, e.g. "Anthropic/hh-rlhf"
        split:        HuggingFace split name, default "train"

    Yields:
        dict with keys prompt / chosen / rejected
    """
    if dataset_name not in ADAPTERS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(ADAPTERS.keys())}"
        )

    from datasets import load_dataset

    # streaming=True means HuggingFace never materialises the full
    # dataset — it downloads and decodes one shard at a time.
    ds      = load_dataset(dataset_name, split=split, streaming=True)
    adapter = ADAPTERS[dataset_name]

    for example in ds:
        rec = adapter(example)
        if rec is not None:
            yield rec


def stream_jsonl(path):
    """
    Generator — yields one normalized record dict at a time from a
    local JSONL file that already contains {prompt, chosen, rejected}
    lines (i.e. a file previously produced by the old prepare_dataset).

    This allows mixing HuggingFace sources with local JSONL files
    in the same pipeline without any format conversion.

    Args:
        path: path to a UTF-8 JSONL file

    Yields:
        dict with keys prompt / chosen / rejected
    """
    import json

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("prompt") and rec.get("chosen") and rec.get("rejected"):
                yield rec
# __main__ — backward-compatible CLI
#
# OLD behaviour:  python prepare_dataset.py --dataset X --output out.jsonl
#                 → wrote a JSONL file
#
# NEW behaviour:  the same CLI flags are accepted but --output is now
#                 IGNORED (kept so existing notebook cells don't break).
#                 Records stream directly into tokenize_dataset via
#                 the stream_dataset() API above.
#
# If you genuinely need to write a JSONL file (e.g. for inspection),
# pass --write-jsonl and --output will be honoured.

def main():
    ap = argparse.ArgumentParser(
        description="Stream-prepare a dataset for the RL data loader pipeline."
    )
    ap.add_argument("--dataset", required=True, choices=list(ADAPTERS.keys()))
    ap.add_argument("--split",  default="train")
    ap.add_argument("--output",  default=None,
                    help="Output path (only used with --write-jsonl)")
    ap.add_argument("--write-jsonl", action="store_true",         help="Write normalized records to --output as JSONL "          "(use only for inspection — not needed for training)")
    args = ap.parse_args()

    import json, sys

    n_written = n_skipped_empty = 0

    if args.write_jsonl:
        if not args.output:
            ap.error("--write-jsonl requires --output")
        out = open(args.output, "w", encoding="utf-8")
    else:
        out = None
    for rec in stream_dataset(args.dataset, args.split):
        n_written += 1
        if out:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if out:
        out.close()
        print(f"wrote {n_written} records to {args.output}, " f"skipped {n_skipped_empty}")
    else:
        print(f"streamed {n_written} records "   f"(use --write-jsonl --output <path> to save as JSONL)")


if __name__ == "__main__":
    main()





"""
prepare_dataset.py — Streaming dataset adapters for the post-training
data loader pipeline.
CHANGES FROM ORIGINAL:
    - No longer writes an intermediate JSONL file to disk.
    - Each adapter is now a generator that yields one normalized record
      at a time so the caller (tokenize_dataset.py) can tokenize and
      shard on the fly without ever holding the full dataset in RAM.
    - When run as __main__ it still accepts the same CLI flags as before
      for backward compatibility, but now pipes records directly into
      the tokenizer instead of writing a JSONL file.
Supported datasets:
    Anthropic/hh-rlhf
    HuggingFaceH4/ultrafeedback_binarized
    stanfordnlp/SHP
Adding a new dataset:
    1. Write an adapt_<name>(example) -> dict | None function.
       Return None to skip a record (replaces the try/except pattern).
    2. Register it in ADAPTERS below.
    That is all — no other file needs to change.
"""