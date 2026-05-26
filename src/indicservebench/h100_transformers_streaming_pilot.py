"""Runpod H100 Transformers streaming pilot runner.

This module is importable on CPU machines for tests. It loads PyTorch,
Transformers, and model weights only from the CLI run path.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import os
import random
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    from indicservebench.prompt_schema import (
        PromptSchemaError,
        assert_valid_prompt_records,
        load_prompt_jsonl,
        prompt_messages,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from indicservebench.prompt_schema import (
        PromptSchemaError,
        assert_valid_prompt_records,
        load_prompt_jsonl,
        prompt_messages,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "prompts" / "prototype_prompts.jsonl"
DEFAULT_OUTPUT_ROOT = Path("/workspace/indicservebench_results/h100_transformers_pilot")
DEFAULT_WORKSPACE_ROOT = Path("/workspace")

EXPERIMENT_TYPE = "preliminary_h100_transformers_streaming_pilot"
VALID_EXPERIMENT_LABEL = "h100_sxm_transformers_pilot"
NON_H100_EXPERIMENT_LABEL = "non_h100_environment"
RUNTIME = "huggingface_transformers_streaming"
LANGUAGE_ORDER = ("en", "hi", "ta", "hinglish")
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_DO_SAMPLE = False
DEFAULT_STREAM_TIMEOUT_SECONDS = 300.0
GENERATED_TEXT_PREVIEW_CHARS = 200

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "sarvam": {
        "model_id": "sarvamai/sarvam-30b-fp8",
        "trust_remote_code": True,
        "required_free_gb": 40.0,
    },
    "qwen": {
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        "trust_remote_code": False,
        "required_free_gb": 40.0,
    },
}

RAW_REQUEST_FIELDS = (
    "experiment_id",
    "experiment_type",
    "experiment_label",
    "timestamp_utc",
    "model_id",
    "model_label",
    "runtime",
    "gpu_name",
    "gpu_vram_gb",
    "torch_version",
    "torch_cuda_version",
    "transformers_version",
    "prompt_id",
    "language",
    "raw_user_prompt_text_hash",
    "raw_user_prompt_tokens",
    "formatted_input_tokens",
    "output_tokens",
    "max_new_tokens",
    "do_sample",
    "seed",
    "ttft_ms",
    "total_latency_ms",
    "end_to_end_output_tokens_per_second",
    "generated_text_preview",
    "success",
    "error_type",
    "error_message",
    "warmup_or_measured",
)

SUMMARY_FIELDS = (
    "language",
    "prompt_count",
    "successful_request_count",
    "failed_request_count",
    "mean_raw_user_prompt_tokens",
    "mean_formatted_input_tokens",
    "median_ttft_ms",
    "p90_ttft_ms",
    "p90_note",
    "median_total_latency_ms",
    "mean_output_tokens",
    "mean_end_to_end_output_tokens_per_second",
)

COMPARISON_FIELDS = (
    "language",
    "model_label",
    "model_id",
    "prompt_count",
    "average_raw_user_prompt_tokens",
    "average_formatted_input_tokens",
    "median_ttft_ms",
    "p90_ttft_ms",
    "p90_note",
    "median_total_latency_ms",
    "average_output_tokens",
    "average_end_to_end_tokens_per_second",
)

LIMITATIONS_STATEMENT = (
    "This is a preliminary single-process Python/Transformers streaming pilot on "
    "one H100. It is not a production benchmark, not SGLang/vLLM serving, and no "
    "CUDA/kernel bottleneck or causal tokenizer-latency claim is established."
)


class H100EnvironmentError(RuntimeError):
    """Raised when a run would be mislabeled as an H100 pilot."""


class PromptSuiteError(ValueError):
    """Raised when the frozen natural prompt suite is missing or invalid."""


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def default_experiment_id(model_label: str, mode: str) -> str:
    stamp = utc_now_iso().replace("-", "").replace(":", "").replace("+00:00", "Z")
    return f"{model_label}_h100_{mode}_{stamp}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def set_workspace_cache_env(workspace_root: Path = DEFAULT_WORKSPACE_ROOT) -> dict[str, str]:
    hf_home = workspace_root / "huggingface"
    hf_hub_cache = hf_home / "hub"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_hub_cache)
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_hub_cache.mkdir(parents=True, exist_ok=True)
    return {
        "HF_HOME": os.environ["HF_HOME"],
        "HF_HUB_CACHE": os.environ["HF_HUB_CACHE"],
    }


def validate_h100_environment(gpu_name: str | None, allow_non_h100: bool) -> str:
    if gpu_name and "H100" in gpu_name.upper():
        return VALID_EXPERIMENT_LABEL
    if allow_non_h100:
        return NON_H100_EXPERIMENT_LABEL
    if not gpu_name:
        raise H100EnvironmentError(
            "No CUDA GPU was detected. Use --allow-non-h100 only for explicitly "
            "labeled diagnostic runs."
        )
    raise H100EnvironmentError(
        f"Detected GPU {gpu_name!r}, not an H100. Use --allow-non-h100 only if "
        f"the output should be labeled {NON_H100_EXPERIMENT_LABEL!r}."
    )


def workspace_disk_stats(path: Path) -> dict[str, float | str]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_gb": round(usage.total / (1024**3), 2),
        "used_gb": round(usage.used / (1024**3), 2),
        "free_gb": round(usage.free / (1024**3), 2),
    }


def ensure_workspace_free_gb(stats: dict[str, float | str], required_free_gb: float) -> None:
    free_gb = float(stats["free_gb"])
    if free_gb < required_free_gb:
        raise RuntimeError(
            f"Insufficient /workspace free disk: {free_gb:.2f} GB available, "
            f"{required_free_gb:.2f} GB required for this model plus headroom."
        )


def load_natural_prompt_suite(path: Path = DEFAULT_PROMPTS_PATH) -> list[dict[str, Any]]:
    try:
        records = assert_valid_prompt_records(load_prompt_jsonl(path))
    except (OSError, PromptSchemaError) as exc:
        raise PromptSuiteError(f"Could not load valid prompt suite from {path}: {exc}") from exc

    natural = [
        record
        for record in records
        if record["suite_type"] == "natural" and record["workload_type"] == "short_128"
    ]
    by_language = {language: 0 for language in LANGUAGE_ORDER}
    for record in natural:
        if record["language"] in by_language:
            by_language[record["language"]] += 1

    if len(natural) != 48 or any(count != 12 for count in by_language.values()):
        raise PromptSuiteError(
            "Expected frozen natural short prompt suite with 48 rows: "
            f"12 each for {', '.join(LANGUAGE_ORDER)}. Found {by_language}."
        )
    return natural


def select_smoke_prompts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for language in LANGUAGE_ORDER:
        candidates = sorted(
            (record for record in records if record["language"] == language),
            key=lambda record: (record["parallel_group_id"], record["prompt_id"]),
        )
        if not candidates:
            raise PromptSuiteError(f"Missing smoke prompt for language {language}")
        selected.append(candidates[0])
    return selected


def select_pilot_prompts(records: list[dict[str, Any]], seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    selected = list(records)
    random.Random(seed).shuffle(selected)
    return selected


def build_warmup_request(records: list[dict[str, Any]]) -> dict[str, Any]:
    warmup_prompt = select_smoke_prompts(records)[0]
    return {"prompt": warmup_prompt, "warmup_or_measured": "warmup"}


def build_requests(
    records: list[dict[str, Any]],
    *,
    smoke: bool,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    measured_prompts = select_smoke_prompts(records) if smoke else select_pilot_prompts(records, seed)
    requests = [build_warmup_request(records)]
    requests.extend(
        {"prompt": record, "warmup_or_measured": "measured"} for record in measured_prompts
    )
    return requests


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def median(values: list[float]) -> float | None:
    return quantile(values, 0.50)


def round_or_none(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def token_count_from_inputs(inputs: Any) -> int:
    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    shape = getattr(input_ids, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[1])
    if isinstance(input_ids, list):
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)
    raise ValueError(f"unsupported input_ids shape/type: {type(input_ids).__name__}")


def output_token_count(outputs: Any, formatted_input_tokens: int) -> int | None:
    sequences = getattr(outputs, "sequences", outputs)
    shape = getattr(sequences, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    return max(0, int(shape[1]) - formatted_input_tokens)


def normalize_preview(text: str) -> str:
    return " ".join(text.split())[:GENERATED_TEXT_PREVIEW_CHARS]


def validate_raw_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in RAW_REQUEST_FIELDS:
        if field not in record:
            errors.append(f"missing field '{field}'")
    if errors:
        return errors

    if record["experiment_type"] != EXPERIMENT_TYPE:
        errors.append(f"experiment_type must be {EXPERIMENT_TYPE}")
    if record["experiment_label"] not in {VALID_EXPERIMENT_LABEL, NON_H100_EXPERIMENT_LABEL}:
        errors.append("experiment_label must identify H100 or explicit non-H100 override")
    if record["model_label"] not in MODEL_CONFIGS:
        errors.append("model_label must be sarvam or qwen")
    if record["model_id"] != MODEL_CONFIGS.get(record["model_label"], {}).get("model_id"):
        errors.append("model_id does not match model_label")
    if record["runtime"] != RUNTIME:
        errors.append(f"runtime must be {RUNTIME}")
    if record["language"] not in LANGUAGE_ORDER:
        errors.append(f"unsupported language {record['language']!r}")
    if record["warmup_or_measured"] not in {"warmup", "measured"}:
        errors.append("warmup_or_measured must be warmup or measured")
    if not isinstance(record["success"], bool):
        errors.append("success must be boolean")
    if not isinstance(record["do_sample"], bool):
        errors.append("do_sample must be boolean")

    nullable_numbers = (
        "gpu_vram_gb",
        "raw_user_prompt_tokens",
        "formatted_input_tokens",
        "output_tokens",
        "ttft_ms",
        "total_latency_ms",
        "end_to_end_output_tokens_per_second",
    )
    for field in nullable_numbers:
        value = record[field]
        if value is not None and not isinstance(value, (int, float)):
            errors.append(f"{field} must be numeric or null")

    for field in ("error_type", "error_message"):
        if record[field] is not None and not isinstance(record[field], str):
            errors.append(f"{field} must be a string or null")

    return errors


def make_raw_record(
    *,
    experiment_id: str,
    experiment_label: str,
    model_id: str,
    model_label: str,
    gpu_name: str | None,
    gpu_vram_gb: float | None,
    torch_version: str | None,
    torch_cuda_version: str | None,
    transformers_version: str | None,
    prompt_record: dict[str, Any],
    raw_user_prompt_text_hash: str,
    raw_user_prompt_tokens: int | None,
    formatted_input_tokens: int | None,
    output_tokens: int | None,
    max_new_tokens: int,
    do_sample: bool,
    seed: int,
    ttft_ms: float | None,
    total_latency_ms: float | None,
    end_to_end_output_tokens_per_second: float | None,
    generated_text_preview: str | None,
    success: bool,
    error_type: str | None,
    error_message: str | None,
    warmup_or_measured: str,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_label": experiment_label,
        "timestamp_utc": utc_now_iso(),
        "model_id": model_id,
        "model_label": model_label,
        "runtime": RUNTIME,
        "gpu_name": gpu_name,
        "gpu_vram_gb": gpu_vram_gb,
        "torch_version": torch_version,
        "torch_cuda_version": torch_cuda_version,
        "transformers_version": transformers_version,
        "prompt_id": prompt_record["prompt_id"],
        "language": prompt_record["language"],
        "raw_user_prompt_text_hash": raw_user_prompt_text_hash,
        "raw_user_prompt_tokens": raw_user_prompt_tokens,
        "formatted_input_tokens": formatted_input_tokens,
        "output_tokens": output_tokens,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "seed": seed,
        "ttft_ms": ttft_ms,
        "total_latency_ms": total_latency_ms,
        "end_to_end_output_tokens_per_second": end_to_end_output_tokens_per_second,
        "generated_text_preview": generated_text_preview,
        "success": success,
        "error_type": error_type,
        "error_message": error_message,
        "warmup_or_measured": warmup_or_measured,
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    errors = validate_raw_record(record)
    if errors:
        raise ValueError("; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    measured = [record for record in records if record["warmup_or_measured"] == "measured"]
    for language in LANGUAGE_ORDER:
        language_records = [record for record in measured if record["language"] == language]
        successful = [record for record in language_records if record["success"]]
        ttft_values = [float(record["ttft_ms"]) for record in successful if record["ttft_ms"] is not None]
        total_values = [
            float(record["total_latency_ms"])
            for record in successful
            if record["total_latency_ms"] is not None
        ]
        raw_tokens = [
            float(record["raw_user_prompt_tokens"])
            for record in successful
            if record["raw_user_prompt_tokens"] is not None
        ]
        formatted_tokens = [
            float(record["formatted_input_tokens"])
            for record in successful
            if record["formatted_input_tokens"] is not None
        ]
        output_tokens = [
            float(record["output_tokens"])
            for record in successful
            if record["output_tokens"] is not None
        ]
        tok_s = [
            float(record["end_to_end_output_tokens_per_second"])
            for record in successful
            if record["end_to_end_output_tokens_per_second"] is not None
        ]
        rows.append(
            {
                "language": language,
                "prompt_count": len(language_records),
                "successful_request_count": len(successful),
                "failed_request_count": len(language_records) - len(successful),
                "mean_raw_user_prompt_tokens": round_or_none(mean(raw_tokens), 3),
                "mean_formatted_input_tokens": round_or_none(mean(formatted_tokens), 3),
                "median_ttft_ms": round_or_none(median(ttft_values), 3),
                "p90_ttft_ms": round_or_none(quantile(ttft_values, 0.90), 3),
                "p90_note": "exploratory; n<20" if 0 < len(ttft_values) < 20 else "",
                "median_total_latency_ms": round_or_none(median(total_values), 3),
                "mean_output_tokens": round_or_none(mean(output_tokens), 3),
                "mean_end_to_end_output_tokens_per_second": round_or_none(mean(tok_s), 3),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})


def markdown_value(value: Any) -> str:
    if value is None:
        return "None"
    if value == "":
        return ""
    return str(value)


def render_human_summary(
    *,
    model_label: str,
    model_id: str,
    mode: str,
    summary_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# H100 Transformers Streaming Pilot",
        "",
        LIMITATIONS_STATEMENT,
        "",
        f"- Model: `{model_label}` (`{model_id}`)",
        f"- Mode: `{mode}`",
        f"- TTFT definition: milliseconds from generation start to first non-empty generated text chunk observed by the Python `TextIteratorStreamer` client.",
        "",
        "| Language | n | successes | mean raw toks | mean formatted toks | median TTFT ms | p90 TTFT ms | median total ms | mean output toks | mean tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {language} | {prompt_count} | {successful_request_count} | "
            "{mean_raw_user_prompt_tokens} | {mean_formatted_input_tokens} | "
            "{median_ttft_ms} | {p90_ttft_ms} | {median_total_latency_ms} | "
            "{mean_output_tokens} | {mean_end_to_end_output_tokens_per_second} |".format(
                **{key: markdown_value(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: compare directions only. Reproducing a pattern here does not prove a tokenizer bottleneck. A production-serving conclusion requires a serving-runtime study such as SGLang/vLLM under a validated supported configuration. No CUDA/kernel bottleneck is claimed.",
        ]
    )
    return "\n".join(lines) + "\n"


def summary_by_model(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for row in summarize_records(records):
        summary[row["language"]] = row
    return summary


def build_comparison_rows(
    sarvam_records: list[dict[str, Any]],
    qwen_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for records in (sarvam_records, qwen_records):
        measured_success = [
            record
            for record in records
            if record["warmup_or_measured"] == "measured" and record["success"]
        ]
        model_label = measured_success[0]["model_label"] if measured_success else records[0]["model_label"]
        model_id = measured_success[0]["model_id"] if measured_success else records[0]["model_id"]
        by_language = summary_by_model(records)
        for language in LANGUAGE_ORDER:
            row = by_language[language]
            rows.append(
                {
                    "language": language,
                    "model_label": model_label,
                    "model_id": model_id,
                    "prompt_count": row["successful_request_count"],
                    "average_raw_user_prompt_tokens": row["mean_raw_user_prompt_tokens"],
                    "average_formatted_input_tokens": row["mean_formatted_input_tokens"],
                    "median_ttft_ms": row["median_ttft_ms"],
                    "p90_ttft_ms": row["p90_ttft_ms"],
                    "p90_note": row["p90_note"],
                    "median_total_latency_ms": row["median_total_latency_ms"],
                    "average_output_tokens": row["mean_output_tokens"],
                    "average_end_to_end_tokens_per_second": row[
                        "mean_end_to_end_output_tokens_per_second"
                    ],
                }
            )
    return rows


def render_comparison_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# H100 Transformers Streaming Pilot Comparison",
        "",
        LIMITATIONS_STATEMENT,
        "",
        "| Language | Model | successful prompts | avg raw toks | avg formatted toks | median TTFT ms | p90 TTFT ms | median total ms | avg output toks | avg tok/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {language} | {model_label} | {prompt_count} | "
            "{average_raw_user_prompt_tokens} | {average_formatted_input_tokens} | "
            "{median_ttft_ms} | {p90_ttft_ms} | {median_total_latency_ms} | "
            "{average_output_tokens} | {average_end_to_end_tokens_per_second} |".format(
                **{key: markdown_value(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "Meeting wording: In this preliminary Transformers streaming pilot on one H100, compare only median TTFT and total-latency direction by language. This observation can be described as consistent or inconsistent with the earlier tokenizer-count hypothesis, but it does not establish causality.",
            "",
            "A production-serving conclusion requires a serving-runtime study such as SGLang/vLLM under a validated supported configuration. No CUDA/kernel bottleneck is claimed.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_run_outputs(run_dir: Path, records: list[dict[str, Any]], model_label: str, model_id: str, mode: str) -> None:
    summary_rows = summarize_records(records)
    write_csv(run_dir / "summary_by_language.csv", summary_rows, SUMMARY_FIELDS)
    (run_dir / "human_readable_summary.md").write_text(
        render_human_summary(
            model_label=model_label,
            model_id=model_id,
            mode=mode,
            summary_rows=summary_rows,
        ),
        encoding="utf-8",
    )


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def runtime_versions(torch_module: Any, transformers_module: Any) -> dict[str, Any]:
    return {
        "torch_version": getattr(torch_module, "__version__", None),
        "torch_cuda_version": getattr(getattr(torch_module, "version", None), "cuda", None),
        "transformers_version": getattr(transformers_module, "__version__", None),
    }


def gpu_metadata(torch_module: Any) -> dict[str, Any]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return {"gpu_name": None, "gpu_vram_gb": None, "cuda_available": False}
    props = cuda.get_device_properties(0)
    return {
        "gpu_name": cuda.get_device_name(0),
        "gpu_vram_gb": round(float(props.total_memory) / (1024**3), 2),
        "cuda_available": True,
    }


def load_runtime_modules() -> tuple[Any, Any, Any, Any, Any]:
    import torch  # type: ignore
    import transformers  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer  # type: ignore

    return torch, transformers, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def parse_device_map(value: str) -> Any:
    if value == "cuda":
        return {"": "cuda:0"}
    return value


def load_model_and_tokenizer(
    *,
    AutoModelForCausalLM: Any,
    AutoTokenizer: Any,
    model_label: str,
    dtype: str,
    device_map: str,
    local_files_only: bool,
) -> tuple[Any, Any]:
    config = MODEL_CONFIGS[model_label]
    model_id = config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=bool(config["trust_remote_code"]),
        local_files_only=local_files_only,
    )
    model_kwargs = {
        "trust_remote_code": bool(config["trust_remote_code"]),
        "device_map": parse_device_map(device_map),
        "local_files_only": local_files_only,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **model_kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, **model_kwargs
        )
    model.eval()
    return model, tokenizer


def seed_runtime(torch_module: Any, seed: int) -> None:
    random.seed(seed)
    if hasattr(torch_module, "manual_seed"):
        torch_module.manual_seed(seed)
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and hasattr(cuda, "manual_seed_all"):
        cuda.manual_seed_all(seed)


def formatted_prompt_text(tokenizer: Any, record: dict[str, Any]) -> str:
    messages = prompt_messages(record)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def infer_model_device(model: Any) -> Any:
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return model_device
    return next(model.parameters()).device


def prepare_inputs(tokenizer: Any, formatted_text: str, device: Any) -> tuple[dict[str, Any], int]:
    inputs = tokenizer(formatted_text, return_tensors="pt")
    formatted_input_tokens = token_count_from_inputs(inputs)
    inputs_dict = dict(inputs)
    inputs_dict.pop("token_type_ids", None)
    moved = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs_dict.items()
    }
    return moved, formatted_input_tokens


def stream_one_request(
    *,
    torch_module: Any,
    TextIteratorStreamer: Any,
    model: Any,
    tokenizer: Any,
    prompt_record: dict[str, Any],
    max_new_tokens: int,
    do_sample: bool,
    stream_timeout_seconds: float,
) -> dict[str, Any]:
    raw_text_hash = sha256_text(prompt_record["user_prompt"])
    raw_tokens = len(tokenizer.encode(prompt_record["user_prompt"], add_special_tokens=False))
    formatted_text = formatted_prompt_text(tokenizer, prompt_record)
    inputs, formatted_input_tokens = prepare_inputs(
        tokenizer, formatted_text, infer_model_device(model)
    )

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=stream_timeout_seconds,
    )
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "streamer": streamer,
    }
    pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    holder: dict[str, Any] = {}

    def generate_target() -> None:
        try:
            with torch_module.no_grad():
                holder["outputs"] = model.generate(**generation_kwargs)
        except BaseException as exc:  # noqa: BLE001 - preserve exact model/runtime error
            holder["error"] = exc
            try:
                streamer.on_finalized_text("", stream_end=True)
            except Exception:
                pass

    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.synchronize()
    start_ns = time.perf_counter_ns()
    thread = threading.Thread(target=generate_target, daemon=True)
    thread.start()

    chunks: list[str] = []
    first_chunk_ns: int | None = None
    try:
        for chunk in streamer:
            if chunk:
                if first_chunk_ns is None:
                    if cuda is not None and cuda.is_available():
                        cuda.synchronize()
                    first_chunk_ns = time.perf_counter_ns()
                chunks.append(chunk)
    except Exception as exc:  # streamer timeout/errors are meaningful request failures
        holder["stream_error"] = exc

    thread.join(timeout=stream_timeout_seconds + 5)
    if thread.is_alive():
        raise TimeoutError("generation thread did not finish before timeout")
    if "error" in holder:
        raise holder["error"]
    if "stream_error" in holder:
        raise holder["stream_error"]

    if cuda is not None and cuda.is_available():
        cuda.synchronize()
    end_ns = time.perf_counter_ns()
    generated_text = "".join(chunks)
    if not generated_text.strip():
        raise RuntimeError("streaming generation completed but produced empty text")

    output_tokens = output_token_count(holder.get("outputs"), formatted_input_tokens)
    if output_tokens is None:
        output_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))

    total_latency_ms = (end_ns - start_ns) / 1_000_000
    ttft_ms = None if first_chunk_ns is None else (first_chunk_ns - start_ns) / 1_000_000
    tok_s = output_tokens / (total_latency_ms / 1000) if total_latency_ms > 0 else None

    return {
        "raw_user_prompt_text_hash": raw_text_hash,
        "raw_user_prompt_tokens": raw_tokens,
        "formatted_input_tokens": formatted_input_tokens,
        "output_tokens": output_tokens,
        "ttft_ms": ttft_ms,
        "total_latency_ms": total_latency_ms,
        "end_to_end_output_tokens_per_second": tok_s,
        "generated_text_preview": normalize_preview(generated_text),
    }


def failure_measurement(tokenizer: Any | None, prompt_record: dict[str, Any]) -> dict[str, Any]:
    raw_tokens: int | None = None
    if tokenizer is not None:
        try:
            raw_tokens = len(tokenizer.encode(prompt_record["user_prompt"], add_special_tokens=False))
        except Exception:
            raw_tokens = None
    return {
        "raw_user_prompt_text_hash": sha256_text(prompt_record["user_prompt"]),
        "raw_user_prompt_tokens": raw_tokens,
        "formatted_input_tokens": None,
        "output_tokens": None,
        "ttft_ms": None,
        "total_latency_ms": None,
        "end_to_end_output_tokens_per_second": None,
        "generated_text_preview": None,
    }


def run_request_to_record(
    *,
    torch_module: Any,
    TextIteratorStreamer: Any,
    model: Any,
    tokenizer: Any,
    request: dict[str, Any],
    experiment_id: str,
    experiment_label: str,
    model_label: str,
    model_id: str,
    gpu: dict[str, Any],
    versions: dict[str, Any],
    max_new_tokens: int,
    do_sample: bool,
    seed: int,
    stream_timeout_seconds: float,
) -> dict[str, Any]:
    prompt_record = request["prompt"]
    try:
        measurement = stream_one_request(
            torch_module=torch_module,
            TextIteratorStreamer=TextIteratorStreamer,
            model=model,
            tokenizer=tokenizer,
            prompt_record=prompt_record,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            stream_timeout_seconds=stream_timeout_seconds,
        )
        success = True
        error_type = None
        error_message = None
    except Exception as exc:  # noqa: BLE001 - errors must be logged verbatim
        measurement = failure_measurement(tokenizer, prompt_record)
        success = False
        error_type = type(exc).__name__
        error_message = str(exc)

    return make_raw_record(
        experiment_id=experiment_id,
        experiment_label=experiment_label,
        model_id=model_id,
        model_label=model_label,
        gpu_name=gpu["gpu_name"],
        gpu_vram_gb=gpu["gpu_vram_gb"],
        torch_version=versions["torch_version"],
        torch_cuda_version=versions["torch_cuda_version"],
        transformers_version=versions["transformers_version"],
        prompt_record=prompt_record,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        seed=seed,
        success=success,
        error_type=error_type,
        error_message=error_message,
        warmup_or_measured=request["warmup_or_measured"],
        **measurement,
    )


def cleanup_model(torch_module: Any, model: Any | None) -> None:
    if model is not None:
        del model
    gc.collect()
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.empty_cache()
        cuda.ipc_collect()


def compare_runs(
    *,
    sarvam_run_dir: Path,
    qwen_run_dir: Path,
    output_dir: Path,
) -> None:
    sarvam_records = read_jsonl(sarvam_run_dir / "raw_requests.jsonl")
    qwen_records = read_jsonl(qwen_run_dir / "raw_requests.jsonl")
    rows = build_comparison_rows(sarvam_records, qwen_records)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "comparison_summary.csv", rows, COMPARISON_FIELDS)
    (output_dir / "comparison_summary.md").write_text(
        render_comparison_summary(rows),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a preliminary H100 Transformers streaming pilot."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Run one measured prompt per language.")
    mode.add_argument("--pilot", action="store_true", help="Run the 48-prompt pilot.")
    mode.add_argument("--compare", action="store_true", help="Compare completed Sarvam and Qwen runs.")

    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="sarvam")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--do-sample", action="store_true", default=DEFAULT_DO_SAMPLE)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--device-map",
        choices=("cuda", "auto"),
        default="cuda",
        help="cuda forces the entire model to cuda:0; auto may offload and is diagnostic only.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-non-h100", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=None)
    parser.add_argument(
        "--stream-timeout-seconds",
        type=float,
        default=DEFAULT_STREAM_TIMEOUT_SECONDS,
    )
    parser.add_argument("--sarvam-run-dir", type=Path, default=None)
    parser.add_argument("--qwen-run-dir", type=Path, default=None)
    return parser.parse_args()


def run_model_mode(args: argparse.Namespace) -> int:
    mode = "smoke" if args.smoke else "pilot"
    model_label = args.model
    model_id = MODEL_CONFIGS[model_label]["model_id"]
    experiment_id = args.experiment_id or default_experiment_id(model_label, mode)
    run_dir = args.output_root / experiment_id
    raw_path = run_dir / "raw_requests.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path.touch(exist_ok=True)

    cache_env = set_workspace_cache_env(args.workspace_root)
    disk_before = workspace_disk_stats(args.workspace_root)
    required_free_gb = args.min_free_gb
    if required_free_gb is None:
        required_free_gb = float(MODEL_CONFIGS[model_label]["required_free_gb"])
    ensure_workspace_free_gb(disk_before, required_free_gb)

    torch_module, transformers_module, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer = (
        load_runtime_modules()
    )
    versions = runtime_versions(torch_module, transformers_module)
    gpu = gpu_metadata(torch_module)
    experiment_label = validate_h100_environment(
        gpu["gpu_name"], allow_non_h100=args.allow_non_h100
    )
    seed_runtime(torch_module, args.seed)

    metadata: dict[str, Any] = {
        "experiment_id": experiment_id,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_label": experiment_label,
        "mode": mode,
        "model_label": model_label,
        "model_id": model_id,
        "runtime": RUNTIME,
        "command": " ".join(sys.argv),
        "start_timestamp_utc": utc_now_iso(),
        "end_timestamp_utc": None,
        "environment_variables": cache_env,
        "workspace_disk_before": disk_before,
        "workspace_disk_after": None,
        "gpu": gpu,
        "dependency_versions": versions,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "seed": args.seed,
        "device_map": args.device_map,
        "dtype": args.dtype,
        "limitations": LIMITATIONS_STATEMENT,
        "fatal_error": None,
    }
    write_metadata(run_dir / "metadata.json", metadata)

    records: list[dict[str, Any]] = []
    model = None
    try:
        prompts = load_natural_prompt_suite(args.prompts)
        requests = build_requests(prompts, smoke=args.smoke, seed=args.seed)
        print(f"Loading {model_id}", flush=True)
        model, tokenizer = load_model_and_tokenizer(
            AutoModelForCausalLM=AutoModelForCausalLM,
            AutoTokenizer=AutoTokenizer,
            model_label=model_label,
            dtype=args.dtype,
            device_map=args.device_map,
            local_files_only=args.local_files_only,
        )

        for request in requests:
            row = run_request_to_record(
                torch_module=torch_module,
                TextIteratorStreamer=TextIteratorStreamer,
                model=model,
                tokenizer=tokenizer,
                request=request,
                experiment_id=experiment_id,
                experiment_label=experiment_label,
                model_label=model_label,
                model_id=model_id,
                gpu=gpu,
                versions=versions,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                seed=args.seed,
                stream_timeout_seconds=args.stream_timeout_seconds,
            )
            append_jsonl(raw_path, row)
            records.append(row)
            print(
                f"{row['warmup_or_measured']} {row['prompt_id']} {row['language']} "
                f"{row['error_type'] if not row['success'] else 'ok'} "
                f"ttft {row['ttft_ms']}",
                flush=True,
            )
            if not row["success"]:
                break

    except Exception as exc:  # noqa: BLE001 - preserve setup/load failures
        metadata["fatal_error"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        cleanup_model(torch_module, model)
        if raw_path.exists():
            records = read_jsonl(raw_path)
            write_run_outputs(run_dir, records, model_label, model_id, mode)
        metadata["end_timestamp_utc"] = utc_now_iso()
        metadata["workspace_disk_after"] = workspace_disk_stats(args.workspace_root)
        write_metadata(run_dir / "metadata.json", metadata)

    if metadata["fatal_error"] is not None:
        return 1
    measured_successes = [
        row for row in records if row["warmup_or_measured"] == "measured" and row["success"]
    ]
    expected_measured = 4 if args.smoke else 48
    if len(measured_successes) != expected_measured:
        return 1
    print(f"DONE: {run_dir}", flush=True)
    return 0


def run_compare_mode(args: argparse.Namespace) -> int:
    if args.sarvam_run_dir is None or args.qwen_run_dir is None:
        raise SystemExit("--compare requires --sarvam-run-dir and --qwen-run-dir")
    experiment_id = args.experiment_id or default_experiment_id("sarvam_qwen", "comparison")
    output_dir = args.output_root / experiment_id
    compare_runs(
        sarvam_run_dir=args.sarvam_run_dir,
        qwen_run_dir=args.qwen_run_dir,
        output_dir=output_dir,
    )
    print(f"DONE: {output_dir}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.compare:
        return run_compare_mode(args)
    return run_model_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
