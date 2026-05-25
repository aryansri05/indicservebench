"""CPU-only tokenizer diagnostics for IndicServeBench.

The script loads tokenizer/config files and chat templates where available. It
must not load model weights, launch serving runtimes, or run GPU benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    from indicservebench.prompt_schema import (
        assert_valid_prompt_records,
        load_prompt_jsonl,
        prompt_messages,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from indicservebench.prompt_schema import (
        assert_valid_prompt_records,
        load_prompt_jsonl,
        prompt_messages,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_NOTE = (
    "This diagnostic identifies tokenizer and chat-template differences. "
    "Actual serving latency will be measured later on GPU."
)

RAW_RESULT_FIELDS = (
    "experiment_id",
    "generated_at_utc",
    "model_id",
    "prompt_id",
    "parallel_group_id",
    "language",
    "script",
    "suite_type",
    "workload_type",
    "intent_category",
    "character_count",
    "raw_user_prompt_tokens",
    "formatted_input_tokens",
    "template_overhead_tokens",
    "raw_tokens_per_character",
    "formatted_tokens_per_character",
    "tokenizer_load_success",
    "raw_tokenization_success",
    "template_success",
    "chat_template_present",
    "error_type",
    "error_message",
    "raw_error_message",
    "template_error_message",
)

SUMMARY_FIELDS = (
    "experiment_id",
    "generated_at_utc",
    "model_id",
    "language",
    "prompt_count",
    "mean_raw_user_prompt_tokens",
    "mean_template_overhead_tokens",
    "mean_formatted_input_tokens",
    "mean_raw_tokens_per_character",
    "mean_formatted_tokens_per_character",
    "min_formatted_input_tokens",
    "max_formatted_input_tokens",
    "failures",
    "error_types",
)


@dataclass(frozen=True)
class ExperimentPaths:
    output_dir: Path
    raw_jsonl: Path
    summary_csv: Path
    metadata_json: Path


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def safe_timestamp_for_id(timestamp: str) -> str:
    return (
        timestamp.replace("-", "")
        .replace(":", "")
        .replace("+", "Z")
        .replace(".", "")
    )


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


def token_count_from_tokenized(tokenized: Any) -> tuple[int | None, str | None]:
    if isinstance(tokenized, dict):
        tokenized = tokenized.get("input_ids")
    elif hasattr(tokenized, "get"):
        tokenized = tokenized.get("input_ids")
    elif hasattr(tokenized, "input_ids"):
        tokenized = tokenized.input_ids

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


def count_raw_user_prompt_tokens(
    tokenizer: Any, record: dict[str, Any]
) -> tuple[int | None, str | None]:
    try:
        tokenized = tokenizer(
            record["user_prompt"],
            add_special_tokens=False,
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    return token_count_from_tokenized(tokenized)


def count_chat_template_tokens(
    tokenizer: Any, record: dict[str, Any]
) -> tuple[int | None, str | None]:
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

    return token_count_from_tokenized(tokenized)


def row_has_failure(row: dict[str, Any]) -> bool:
    return not (
        row["tokenizer_load_success"]
        and row["raw_tokenization_success"]
        and row["template_success"]
    )


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
                raw_user_prompt_tokens=None,
                formatted_input_tokens=None,
                tokenizer_load_success=False,
                raw_tokenization_success=False,
                template_success=False,
                chat_template_present=False,
                error_type=error_type,
                error_message=error_message,
                raw_error_message=error_message,
                template_error_message=error_message,
            )
        )
    return rows


def build_row(
    *,
    experiment_id: str,
    generated_at_utc: str,
    model_id: str,
    record: dict[str, Any],
    raw_user_prompt_tokens: int | None,
    formatted_input_tokens: int | None,
    tokenizer_load_success: bool,
    raw_tokenization_success: bool,
    template_success: bool,
    chat_template_present: bool,
    error_type: str | None,
    error_message: str | None,
    raw_error_message: str | None,
    template_error_message: str | None,
) -> dict[str, Any]:
    character_count = len(record["user_prompt"])

    template_overhead_tokens = None
    if raw_user_prompt_tokens is not None and formatted_input_tokens is not None:
        template_overhead_tokens = formatted_input_tokens - raw_user_prompt_tokens

    raw_tokens_per_character = None
    if raw_user_prompt_tokens is not None and character_count > 0:
        raw_tokens_per_character = raw_user_prompt_tokens / character_count

    formatted_tokens_per_character = None
    if formatted_input_tokens is not None and character_count > 0:
        formatted_tokens_per_character = formatted_input_tokens / character_count

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
        "raw_user_prompt_tokens": raw_user_prompt_tokens,
        "formatted_input_tokens": formatted_input_tokens,
        "template_overhead_tokens": template_overhead_tokens,
        "raw_tokens_per_character": raw_tokens_per_character,
        "formatted_tokens_per_character": formatted_tokens_per_character,
        "tokenizer_load_success": tokenizer_load_success,
        "raw_tokenization_success": raw_tokenization_success,
        "template_success": template_success,
        "chat_template_present": chat_template_present,
        "error_type": error_type,
        "error_message": error_message,
        "raw_error_message": raw_error_message,
        "template_error_message": template_error_message,
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
        raw_user_prompt_tokens, raw_error = count_raw_user_prompt_tokens(tokenizer, record)
        formatted_input_tokens, template_error = count_chat_template_tokens(tokenizer, record)

        raw_success = raw_user_prompt_tokens is not None
        template_success = formatted_input_tokens is not None
        error_types: list[str] = []
        error_messages: list[str] = []
        if not raw_success:
            error_types.append("raw_tokenization_failed")
            error_messages.append(raw_error or "raw tokenization failed")
        if not template_success:
            error_types.append("template_failed")
            error_messages.append(template_error or "chat template failed")

        rows.append(
            build_row(
                experiment_id=experiment_id,
                generated_at_utc=generated_at_utc,
                model_id=model_id,
                record=record,
                raw_user_prompt_tokens=raw_user_prompt_tokens,
                formatted_input_tokens=formatted_input_tokens,
                tokenizer_load_success=True,
                raw_tokenization_success=raw_success,
                template_success=template_success,
                chat_template_present=chat_template_present,
                error_type=";".join(error_types) if error_types else None,
                error_message=" | ".join(error_messages) if error_messages else None,
                raw_error_message=raw_error,
                template_error_message=template_error,
            )
        )

    return rows


def prepare_experiment_paths(
    output_root: Path, experiment_id: str, overwrite: bool = False
) -> ExperimentPaths:
    output_dir = output_root / experiment_id
    if output_dir.exists():
        existing_files = [path for path in output_dir.iterdir() if path.is_file()]
        if existing_files and not overwrite:
            raise FileExistsError(
                f"{output_dir} already contains files. Use a new --experiment-id."
            )
        if overwrite:
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    return ExperimentPaths(
        output_dir=output_dir,
        raw_jsonl=output_dir / "raw.jsonl",
        summary_csv=output_dir / "summary.csv",
        metadata_json=output_dir / "metadata.json",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize_rows(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_FIELDS))
        writer.writeheader()
        writer.writerows(summary_rows)


def write_metadata(
    path: Path,
    *,
    experiment_id: str,
    generated_at_utc: str,
    prompts_path: Path,
    models_config_path: Path,
    model_configs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    paths: ExperimentPaths,
) -> None:
    metadata = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at_utc,
        "diagnostic_note": DIAGNOSTIC_NOTE,
        "gpu_or_serving_code_used": False,
        "prompts_path": str(prompts_path),
        "models_config_path": str(models_config_path),
        "model_ids": [model["model_id"] for model in model_configs],
        "prompt_count": len({row["prompt_id"] for row in rows}),
        "row_count": len(rows),
        "all_tokenizers_loaded": all(row["tokenizer_load_success"] for row in rows),
        "all_raw_tokenization_succeeded": all(
            row["raw_tokenization_success"] for row in rows
        ),
        "all_templates_succeeded": all(row["template_success"] for row in rows),
        "raw_result_fields": list(RAW_RESULT_FIELDS),
        "summary_fields": list(SUMMARY_FIELDS),
        "result_files": {
            "raw_jsonl": str(paths.raw_jsonl),
            "summary_csv": str(paths.summary_csv),
            "metadata_json": str(paths.metadata_json),
        },
        "character_count_definition": "len(user_prompt)",
        "raw_user_prompt_tokens_definition": (
            "Token count for user_prompt only, add_special_tokens=False."
        ),
        "formatted_input_tokens_definition": (
            "Token count after tokenizer.apply_chat_template(messages, "
            "tokenize=True, add_generation_prompt=True)."
        ),
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
        raw_counts = [
            float(row["raw_user_prompt_tokens"])
            for row in group_rows
            if row["raw_user_prompt_tokens"] is not None
        ]
        overhead_counts = [
            float(row["template_overhead_tokens"])
            for row in group_rows
            if row["template_overhead_tokens"] is not None
        ]
        formatted_counts = [
            float(row["formatted_input_tokens"])
            for row in group_rows
            if row["formatted_input_tokens"] is not None
        ]
        raw_ratios = [
            float(row["raw_tokens_per_character"])
            for row in group_rows
            if row["raw_tokens_per_character"] is not None
        ]
        formatted_ratios = [
            float(row["formatted_tokens_per_character"])
            for row in group_rows
            if row["formatted_tokens_per_character"] is not None
        ]
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
                "mean_raw_user_prompt_tokens": mean(raw_counts),
                "mean_template_overhead_tokens": mean(overhead_counts),
                "mean_formatted_input_tokens": mean(formatted_counts),
                "mean_raw_tokens_per_character": mean(raw_ratios),
                "mean_formatted_tokens_per_character": mean(formatted_ratios),
                "min_formatted_input_tokens": min(formatted_counts)
                if formatted_counts
                else None,
                "max_formatted_input_tokens": max(formatted_counts)
                if formatted_counts
                else None,
                "failures": sum(1 for row in group_rows if row_has_failure(row)),
                "error_types": ";".join(error_types),
            }
        )

    return summary_rows


def run_analysis(
    *,
    prompts_path: Path,
    models_config_path: Path,
    output_root: Path,
    experiment_id: str,
    trust_remote_code_override: bool = False,
    local_files_only: bool = False,
    overwrite: bool = False,
) -> tuple[ExperimentPaths, list[dict[str, Any]]]:
    generated_at_utc = utc_now_iso()
    paths = prepare_experiment_paths(output_root, experiment_id, overwrite=overwrite)
    records = load_prompt_jsonl(prompts_path)
    assert_valid_prompt_records(records)
    model_configs = load_model_configs(models_config_path)

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
                    trust_remote_code_override=trust_remote_code_override,
                    local_files_only=local_files_only,
                )
            )

    write_jsonl(paths.raw_jsonl, all_rows)
    write_summary_csv(paths.summary_csv, all_rows)
    write_metadata(
        paths.metadata_json,
        experiment_id=experiment_id,
        generated_at_utc=generated_at_utc,
        prompts_path=prompts_path,
        models_config_path=models_config_path,
        model_configs=model_configs,
        rows=all_rows,
        paths=paths,
    )
    return paths, all_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CPU-only tokenizer diagnostic analysis."
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
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "tokenizer",
        help="Root directory for isolated tokenizer experiment outputs.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Optional experiment ID. Defaults to tokenizer_diagnostic timestamp.",
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the experiment directory if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generated_at_utc = utc_now_iso()
    experiment_id = args.experiment_id or (
        "tokenizer_diagnostic_" + safe_timestamp_for_id(generated_at_utc)
    )

    paths, rows = run_analysis(
        prompts_path=args.prompts,
        models_config_path=args.models_config,
        output_root=args.output_root,
        experiment_id=experiment_id,
        trust_remote_code_override=args.trust_remote_code,
        local_files_only=args.local_files_only,
        overwrite=args.overwrite,
    )

    status = {
        "experiment_id": experiment_id,
        "output_dir": str(paths.output_dir),
        "raw_jsonl": str(paths.raw_jsonl),
        "summary_csv": str(paths.summary_csv),
        "metadata_json": str(paths.metadata_json),
        "records_written": len(rows),
        "all_tokenizers_loaded": all(row["tokenizer_load_success"] for row in rows),
        "all_raw_tokenization_succeeded": all(
            row["raw_tokenization_success"] for row in rows
        ),
        "all_templates_succeeded": all(row["template_success"] for row in rows),
        "diagnostic_note": DIAGNOSTIC_NOTE,
    }
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
