"""CPU-only tokenizer analysis for IndicServeBench.

The script loads tokenizer files and chat templates where available. It must not
load model weights or start any serving runtime.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


try:
    from indicservebench.prompt_schema import (
        assert_valid_prompt_records,
        load_prompt_jsonl,
        prompt_character_count,
        prompt_messages,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from indicservebench.prompt_schema import (
        assert_valid_prompt_records,
        load_prompt_jsonl,
        prompt_character_count,
        prompt_messages,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def load_model_configs(config_path: Path) -> list[dict[str, Any]]:
    """Load model configs from YAML, with a small fallback for model IDs."""

    try:
        import yaml  # type: ignore
    except Exception:
        return _load_model_configs_without_yaml(config_path)

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise ValueError(f"{config_path}: expected a top-level 'models' list")

    model_configs: list[dict[str, Any]] = []
    for item in data["models"]:
        if not isinstance(item, dict):
            continue
        model_id = item.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            continue
        tokenizer_cfg = item.get("tokenizer") if isinstance(item.get("tokenizer"), dict) else {}
        model_configs.append(
            {
                "model_id": model_id,
                "short_name": item.get("short_name", model_id),
                "trust_remote_code": bool(tokenizer_cfg.get("trust_remote_code", False)),
            }
        )

    if not model_configs:
        raise ValueError(f"{config_path}: no usable model_id entries found")
    return model_configs


def _load_model_configs_without_yaml(config_path: Path) -> list[dict[str, Any]]:
    """Fallback parser for the simple repository YAML when PyYAML is absent."""

    model_configs: list[dict[str, Any]] = []
    pattern = re.compile(r"^\s*model_id:\s*[\"']?([^\"'\n]+)[\"']?\s*$")
    with config_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.match(line)
            if match:
                model_id = match.group(1).strip()
                model_configs.append(
                    {
                        "model_id": model_id,
                        "short_name": model_id,
                        "trust_remote_code": False,
                    }
                )
    if not model_configs:
        raise ValueError(
            f"{config_path}: PyYAML is not installed and fallback parser found no model_id entries"
        )
    return model_configs


def import_auto_tokenizer() -> tuple[Any | None, str | None]:
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return AutoTokenizer, None


def count_chat_template_tokens(tokenizer: Any, record: dict[str, Any]) -> tuple[int | None, str | None]:
    if not getattr(tokenizer, "chat_template", None):
        return None, "chat_template_missing"

    messages = prompt_messages(record)
    try:
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    if isinstance(tokenized, dict):
        tokenized = tokenized.get("input_ids")

    if hasattr(tokenized, "shape"):
        shape = getattr(tokenized, "shape")
        if len(shape) == 2:
            return int(shape[1]), None
        if len(shape) == 1:
            return int(shape[0]), None

    if isinstance(tokenized, list):
        if tokenized and isinstance(tokenized[0], list):
            return len(tokenized[0]), None
        return len(tokenized), None

    return None, f"unsupported_tokenized_type:{type(tokenized).__name__}"


def failure_rows_for_model(
    *,
    experiment_id: str,
    generated_at_utc: str,
    model_id: str,
    records: list[dict[str, Any]],
    error_type: str,
    error_message: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            build_row(
                experiment_id=experiment_id,
                generated_at_utc=generated_at_utc,
                model_id=model_id,
                record=record,
                formatted_input_tokens=None,
                template_success=False,
                template_status=error_type,
                error_type=error_type,
                error_message=error_message,
                tokenizer_load_success=False,
                chat_template_present=False,
            )
        )
    return rows


def build_row(
    *,
    experiment_id: str,
    generated_at_utc: str,
    model_id: str,
    record: dict[str, Any],
    formatted_input_tokens: int | None,
    template_success: bool,
    template_status: str,
    error_type: str | None,
    error_message: str | None,
    tokenizer_load_success: bool,
    chat_template_present: bool,
) -> dict[str, Any]:
    character_count = prompt_character_count(record)
    tokens_per_character = None
    if formatted_input_tokens is not None and character_count > 0:
        tokens_per_character = formatted_input_tokens / character_count

    return {
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at_utc,
        "model_id": model_id,
        "prompt_id": record["prompt_id"],
        "parallel_group_id": record["parallel_group_id"],
        "language": record["language"],
        "script": record["script"],
        "suite_type": record["suite_type"],
        "workload_type": record["workload_type"],
        "intent_category": record["intent_category"],
        "character_count": character_count,
        "formatted_input_tokens": formatted_input_tokens,
        "tokens_per_character": tokens_per_character,
        "template_success": template_success,
        "template_status": template_status,
        "tokenizer_load_success": tokenizer_load_success,
        "chat_template_present": chat_template_present,
        "error_type": error_type,
        "error_message": error_message,
    }


def analyze_model(
    *,
    AutoTokenizer: Any,
    model_config: dict[str, Any],
    records: list[dict[str, Any]],
    experiment_id: str,
    generated_at_utc: str,
    trust_remote_code_override: bool,
    local_files_only: bool,
) -> list[dict[str, Any]]:
    model_id = model_config["model_id"]
    trust_remote_code = bool(model_config.get("trust_remote_code", False))
    if trust_remote_code_override:
        trust_remote_code = True

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        return failure_rows_for_model(
            experiment_id=experiment_id,
            generated_at_utc=generated_at_utc,
            model_id=model_id,
            records=records,
            error_type="tokenizer_load_failed",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    rows: list[dict[str, Any]] = []
    chat_template_present = bool(getattr(tokenizer, "chat_template", None))

    for record in records:
        formatted_input_tokens, error = count_chat_template_tokens(tokenizer, record)
        if formatted_input_tokens is None:
            rows.append(
                build_row(
                    experiment_id=experiment_id,
                    generated_at_utc=generated_at_utc,
                    model_id=model_id,
                    record=record,
                    formatted_input_tokens=None,
                    template_success=False,
                    template_status="template_failed",
                    error_type="template_failed",
                    error_message=error,
                    tokenizer_load_success=True,
                    chat_template_present=chat_template_present,
                )
            )
        else:
            rows.append(
                build_row(
                    experiment_id=experiment_id,
                    generated_at_utc=generated_at_utc,
                    model_id=model_id,
                    record=record,
                    formatted_input_tokens=formatted_input_tokens,
                    template_success=True,
                    template_status="ok",
                    error_type=None,
                    error_message=None,
                    tokenizer_load_success=True,
                    chat_template_present=chat_template_present,
                )
            )

    return rows


def write_jsonl_append(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize_rows(rows)
    fieldnames = [
        "experiment_id",
        "generated_at_utc",
        "model_id",
        "language",
        "prompt_count",
        "template_success_count",
        "template_failure_count",
        "mean_character_count",
        "mean_formatted_input_tokens",
        "mean_tokens_per_character",
        "min_formatted_input_tokens",
        "max_formatted_input_tokens",
        "error_types",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_id"], row["language"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (model_id, language), group_rows in sorted(grouped.items()):
        successful = [row for row in group_rows if row["template_success"]]
        token_counts = [
            float(row["formatted_input_tokens"])
            for row in successful
            if row["formatted_input_tokens"] is not None
        ]
        token_ratios = [
            float(row["tokens_per_character"])
            for row in successful
            if row["tokens_per_character"] is not None
        ]
        char_counts = [float(row["character_count"]) for row in group_rows]
        error_types = sorted(
            {
                str(row["error_type"])
                for row in group_rows
                if row["error_type"] is not None
            }
        )

        summary_rows.append(
            {
                "experiment_id": group_rows[0]["experiment_id"],
                "generated_at_utc": group_rows[0]["generated_at_utc"],
                "model_id": model_id,
                "language": language,
                "prompt_count": len(group_rows),
                "template_success_count": len(successful),
                "template_failure_count": len(group_rows) - len(successful),
                "mean_character_count": mean(char_counts),
                "mean_formatted_input_tokens": mean(token_counts),
                "mean_tokens_per_character": mean(token_ratios),
                "min_formatted_input_tokens": min(token_counts) if token_counts else None,
                "max_formatted_input_tokens": max(token_counts) if token_counts else None,
                "error_types": ";".join(error_types),
            }
        )

    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPU-only tokenizer analysis.")
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
        "--raw-jsonl",
        type=Path,
        default=PROJECT_ROOT / "results" / "tokenizer" / "tokenizer_raw.jsonl",
        help="Append-only raw JSONL output path.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "tokenizer" / "tokenizer_summary.csv",
        help="CSV summary output path for the current run.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Optional experiment ID. Defaults to tokenizer timestamp.",
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
    generated_at_utc = utc_now_iso()
    experiment_id = args.experiment_id or "tokenizer_" + generated_at_utc.replace(":", "").replace("+", "Z")

    records = load_prompt_jsonl(args.prompts)
    assert_valid_prompt_records(records)
    model_configs = load_model_configs(args.models_config)

    AutoTokenizer, import_error = import_auto_tokenizer()
    all_rows: list[dict[str, Any]] = []

    if AutoTokenizer is None:
        for model_config in model_configs:
            all_rows.extend(
                failure_rows_for_model(
                    experiment_id=experiment_id,
                    generated_at_utc=generated_at_utc,
                    model_id=model_config["model_id"],
                    records=records,
                    error_type="transformers_import_failed",
                    error_message=import_error or "transformers import failed",
                )
            )
    else:
        for model_config in model_configs:
            all_rows.extend(
                analyze_model(
                    AutoTokenizer=AutoTokenizer,
                    model_config=model_config,
                    records=records,
                    experiment_id=experiment_id,
                    generated_at_utc=generated_at_utc,
                    trust_remote_code_override=args.trust_remote_code,
                    local_files_only=args.local_files_only,
                )
            )

    write_jsonl_append(args.raw_jsonl, all_rows)
    write_summary_csv(args.summary_csv, all_rows)

    status = {
        "experiment_id": experiment_id,
        "raw_jsonl": str(args.raw_jsonl),
        "summary_csv": str(args.summary_csv),
        "records_written": len(all_rows),
        "model_count": len(model_configs),
        "prompt_count": len(records),
    }
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
