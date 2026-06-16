# prepare_dataset.py
import json
import re
import argparse
from datasets import load_dataset


def split_on_last_assistant(text):
    idx = text.rfind("\n\nAssistant:")
    prompt = text[:idx].strip()
    response = text[idx + len("\n\nAssistant:"):].strip()
    return prompt, response


def adapt_hh_rlhf(example):
    prompt, chosen_resp = split_on_last_assistant(example["chosen"])
    _, rejected_resp = split_on_last_assistant(example["rejected"])
    return {"prompt": prompt, "chosen": chosen_resp, "rejected": rejected_resp}


def adapt_ultrafeedback(example):
    chosen = example["chosen"]
    rejected = example["rejected"]
    chosen_text = chosen[-1]["content"] if isinstance(chosen, list) else chosen
    rejected_text = rejected[-1]["content"] if isinstance(rejected, list) else rejected
    return {"prompt": example["prompt"], "chosen": chosen_text, "rejected": rejected_text}


def adapt_shp(example):
    if example["labels"] == 1:
        chosen, rejected = example["human_ref_A"], example["human_ref_B"]
    else:
        chosen, rejected = example["human_ref_B"], example["human_ref_A"]
    return {"prompt": example["history"], "chosen": chosen, "rejected": rejected}


ADAPTERS = {
    "Anthropic/hh-rlhf": adapt_hh_rlhf,
    "HuggingFaceH4/ultrafeedback_binarized": adapt_ultrafeedback,
    "stanfordnlp/SHP": adapt_shp,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(ADAPTERS.keys()))
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    adapter = ADAPTERS[args.dataset]

    n_written, n_skipped = 0, 0
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in ds:
            try:
                rec = adapter(ex)
                if not rec["prompt"] or not rec["chosen"] or not rec["rejected"]:
                    n_skipped += 1
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
            except Exception:
                n_skipped += 1

    print(f"wrote {n_written} records, skipped {n_skipped}")


if __name__ == "__main__":
    main()