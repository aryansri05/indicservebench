"""Inspect tokenizer fragments for one Hindi, Tamil, and Hinglish prompt.

This is a CPU-only diagnostic command. It loads tokenizers only and does not
load model weights or start any serving runtime.
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
    from indicservebench.tokenizer_analysis import (
        PROJECT_ROOT,
        import_auto_tokenizer,
        load_model_configs,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from indicservebench.prompt_schema import (
        assert_valid_prompt_records,
        load_prompt_jsonl,
    )
    from indicservebench.tokenizer_analysis import (
        PROJECT_ROOT,
        import_auto_tokenizer,
        load_model_configs,
    )


LANGUAGE_ORDER = ("hi", "ta", "hinglish")


def select_one_prompt_per_language(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for language in LANGUAGE_ORDER:
        candidates = sorted(
            (record for record in records if record["language"] == language),
            key=lambda record: record["prompt_id"],
        )
        if not candidates:
            raise ValueError(f"No prompt found for language '{language}'")
        selected.append(candidates[0])
    return selected


def inspect_fragments(
    *,
    tokenizer: Any,
    model_id: str,
    record: dict[str, Any],
    max_tokens: int | None,
) -> dict[str, Any]:
    token_ids = tokenizer.encode(record["user_prompt"], add_special_tokens=False)
    pieces = [tokenizer.decode([token_id]) for token_id in token_ids]

    display_token_ids = token_ids
    display_pieces = pieces
    truncated = False
    if max_tokens is not None and len(token_ids) > max_tokens:
        display_token_ids = token_ids[:max_tokens]
        display_pieces = pieces[:max_tokens]
        truncated = True

    return {
        "model_id": model_id,
        "language": record["language"],
        "script": record["script"],
        "prompt_id": record["prompt_id"],
        "parallel_group_id": record["parallel_group_id"],
        "text_field": "user_prompt",
        "text": record["user_prompt"],
        "token_count": len(token_ids),
        "displayed_token_count": len(display_token_ids),
        "truncated": truncated,
        "tokens": [
            {"index": index, "token_id": token_id, "piece": piece}
            for index, (token_id, piece) in enumerate(zip(display_token_ids, display_pieces))
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print raw user-prompt token IDs and decoded pieces."
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=PROJECT_ROOT / "prompts" / "prototype_prompts.jsonl",
        help="Prompt JSONL path.",
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "models.yaml",
        help="Model configuration YAML path.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optionally truncate printed token lists for readability.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow tokenizer loading with trust_remote_code=True where needed.",
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
    selected_records = select_one_prompt_per_language(records)
    model_configs = load_model_configs(args.models_config)

    AutoTokenizer, import_error = import_auto_tokenizer()
    if AutoTokenizer is None:
        print(json.dumps({"status": "error", "error": import_error}, indent=2))
        return 1

    outputs: list[dict[str, Any]] = []
    for model_config in model_configs:
        model_id = model_config["model_id"]
        trust_remote_code = bool(model_config.get("trust_remote_code", False))
        if args.trust_remote_code:
            trust_remote_code = True

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                local_files_only=args.local_files_only,
            )
        except Exception as exc:
            outputs.append(
                {
                    "model_id": model_id,
                    "status": "tokenizer_load_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        for record in selected_records:
            try:
                outputs.append(
                    inspect_fragments(
                        tokenizer=tokenizer,
                        model_id=model_id,
                        record=record,
                        max_tokens=args.max_tokens,
                    )
                )
            except Exception as exc:
                outputs.append(
                    {
                        "model_id": model_id,
                        "language": record["language"],
                        "prompt_id": record["prompt_id"],
                        "status": "tokenization_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
