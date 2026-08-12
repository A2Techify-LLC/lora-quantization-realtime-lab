from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-out", type=Path, required=True)
    parser.add_argument("--merged-out", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_out, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_out)
    model = model.merge_and_unload()
    args.merged_out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.merged_out, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_out)


if __name__ == "__main__":
    main()

