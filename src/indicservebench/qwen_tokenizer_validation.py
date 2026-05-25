"""Validate Qwen FP8 and non-FP8 raw tokenizer equivalence.

This is CPU-only. It loads tokenizer files only, compares raw user_prompt token
IDs, and does not load model weights or launch a serving runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


try:
    from indicservebench.prompt_schema import (
        assert_valid_prompt_records,
        load_prompt_jsonl,
    )
    from indicservebench.tokenizer_analysis import PROJECT_ROOT, import_auto_tokenizer
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from indicservebench.prompt_schema import (
        assert_valid_prompt_records,
        load_prompt_jsonl,
    )
    from indicservebench.tokenizer_analysis import PROJECT_ROOT, import_auto_tokenizer


QWEN_FP8_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
QWEN_BASE_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
SARVAM_MODEL_ID = "sarvamai/sarvam-30b"
SELECTED_PROMPT_IDS = ("nat_001_hi", "nat_001_ta", "nat_001_hinglish", "nat_001_en")


def raw_user_prompt_token_ids(tokenizer: Any, record: dict[str, Any]) -> list[int]:
    """Tokenize only record['user_prompt'] with no special tokens."""

    return list(tokenizer.encode(record["user_prompt"], add_special_tokens=False))


def decoded_fragments(tokenizer: Any, token_ids: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "token_id": token_id,
            "piece": tokenizer.decode([token_id]),
        }
        for index, token_id in enumerate(token_ids)
    ]


def load_tokenizer(AutoTokenizer: Any, model_id: str, local_files_only: bool) -> Any:
    return AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)


def compare_qwen_tokenizers(
    *,
    records: list[dict[str, Any]],
    AutoTokenizer: Any,
    local_files_only: bool,
) -> dict[str, Any]:
    fp8_tokenizer = load_tokenizer(AutoTokenizer, QWEN_FP8_MODEL_ID, local_files_only)
    base_tokenizer = load_tokenizer(AutoTokenizer, QWEN_BASE_MODEL_ID, local_files_only)

    differing_prompt_ids: list[str] = []
    comparisons: list[dict[str, Any]] = []
    for record in records:
        fp8_ids = raw_user_prompt_token_ids(fp8_tokenizer, record)
        base_ids = raw_user_prompt_token_ids(base_tokenizer, record)
        same = fp8_ids == base_ids
        if not same:
            differing_prompt_ids.append(record["prompt_id"])
        comparisons.append(
            {
                "prompt_id": record["prompt_id"],
                "language": record["language"],
                "fp8_count": len(fp8_ids),
                "non_fp8_count": len(base_ids),
                "token_ids_identical": same,
            }
        )

    return {
        "fp8_model_id": QWEN_FP8_MODEL_ID,
        "non_fp8_model_id": QWEN_BASE_MODEL_ID,
        "prompt_count": len(records),
        "all_raw_token_ids_match": not differing_prompt_ids,
        "differing_prompt_ids": differing_prompt_ids,
        "comparisons": comparisons,
    }


def selected_fragment_report(
    *,
    records: list[dict[str, Any]],
    AutoTokenizer: Any,
    local_files_only: bool,
) -> list[dict[str, Any]]:
    records_by_id = {record["prompt_id"]: record for record in records}
    tokenizers = {
        SARVAM_MODEL_ID: load_tokenizer(AutoTokenizer, SARVAM_MODEL_ID, local_files_only),
        QWEN_FP8_MODEL_ID: load_tokenizer(AutoTokenizer, QWEN_FP8_MODEL_ID, local_files_only),
        QWEN_BASE_MODEL_ID: load_tokenizer(AutoTokenizer, QWEN_BASE_MODEL_ID, local_files_only),
    }

    report: list[dict[str, Any]] = []
    for prompt_id in SELECTED_PROMPT_IDS:
        record = records_by_id[prompt_id]
        for model_id, tokenizer in tokenizers.items():
            token_ids = raw_user_prompt_token_ids(tokenizer, record)
            report.append(
                {
                    "model_id": model_id,
                    "prompt_id": prompt_id,
                    "language": record["language"],
                    "raw_text": record["user_prompt"],
                    "token_count": len(token_ids),
                    "tokens": decoded_fragments(tokenizer, token_ids),
                }
            )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Qwen FP8 and non-FP8 raw user-prompt tokenization."
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=PROJECT_ROOT / "prompts" / "prototype_prompts.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "tokenizer"
        / "qwen_tokenizer_validation_v1"
        / "validation.json",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download tokenizer/config files from Hugging Face.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_prompt_jsonl(args.prompts)
    assert_valid_prompt_records(records)

    AutoTokenizer, import_error = import_auto_tokenizer()
    if AutoTokenizer is None:
        print(json.dumps({"status": "error", "error": import_error}, indent=2))
        return 1

    comparison = compare_qwen_tokenizers(
        records=records,
        AutoTokenizer=AutoTokenizer,
        local_files_only=args.local_files_only,
    )
    fragments = selected_fragment_report(
        records=records,
        AutoTokenizer=AutoTokenizer,
        local_files_only=args.local_files_only,
    )
    result = {
        "schema_version": "1.0",
        "raw_token_accounting": {
            "valid": True,
            "definition": "Raw token IDs are computed from record['user_prompt'] only with add_special_tokens=False.",
            "excludes": [
                "system_prompt",
                "context_text",
                "role markers",
                "chat-template tokens",
            ],
        },
        "qwen_comparison": comparison,
        "selected_fragment_report": fragments,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "all_raw_token_ids_match": comparison["all_raw_token_ids_match"],
                "differing_prompt_ids": comparison["differing_prompt_ids"],
                "raw_token_accounting_valid": True,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
