"""H100 SGLang OpenAI-compatible streaming pilot client.

This client assumes an SGLang server is already running on the H100 pod. It is
CPU-importable for tests and performs only HTTP client-side measurement.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

try:
    from indicservebench.h100_transformers_streaming_pilot import (
        LANGUAGE_ORDER,
        load_natural_prompt_suite,
        mean,
        median,
        quantile,
        round_or_none,
        sha256_text,
    )
    from indicservebench.prompt_schema import prompt_messages
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from indicservebench.h100_transformers_streaming_pilot import (
        LANGUAGE_ORDER,
        load_natural_prompt_suite,
        mean,
        median,
        quantile,
        round_or_none,
        sha256_text,
    )
    from indicservebench.prompt_schema import prompt_messages


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "prompts" / "prototype_prompts.jsonl"
DEFAULT_OUTPUT_ROOT = Path("/workspace/indicservebench_results/sarvam_h100_sglang_pilot")
DEFAULT_BASE_URL = "http://127.0.0.1:30000"

EXPERIMENT_TYPE = "preliminary_h100_sglang_streaming_pilot"
RUNTIME = "sglang_openai_compatible_streaming"
MODEL_CONFIGS = {
    "sarvam": {
        "model_id": "sarvamai/sarvam-30b-fp8",
        "trust_remote_code": True,
        "title": "Sarvam H100 SGLang Streaming Pilot",
        "meeting_label": "preliminary Sarvam-30B-FP8 H100 SGLang streaming pilot",
    },
    "qwen": {
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        "trust_remote_code": False,
        "title": "Qwen H100 SGLang Streaming Pilot",
        "meeting_label": "preliminary Qwen3-30B-A3B-FP8 H100 SGLang streaming pilot",
    },
}
MODEL_ID = MODEL_CONFIGS["sarvam"]["model_id"]
DEFAULT_SEED = 42
DEFAULT_MAX_TOKENS = 32

RAW_FIELDS = (
    "experiment_id",
    "experiment_type",
    "timestamp_utc",
    "model_id",
    "model_label",
    "runtime",
    "base_url",
    "prompt_id",
    "language",
    "raw_user_prompt_text_hash",
    "raw_user_prompt_tokens",
    "formatted_input_tokens",
    "output_tokens",
    "max_tokens",
    "temperature",
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

CLAIM_BOUNDARY = (
    "This is a preliminary H100 SGLang streaming pilot, single concurrency, "
    "using the frozen natural prompt suite. It is not a production benchmark "
    "and does not establish a CUDA/kernel bottleneck or a causal "
    "tokenizer-latency claim."
)


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def default_experiment_id(model_label: str, mode: str) -> str:
    stamp = utc_now_iso().replace("-", "").replace(":", "").replace("+00:00", "Z")
    return f"{model_label}_h100_sglang_{mode}_{stamp}"


def select_smoke_prompts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for language in LANGUAGE_ORDER:
        candidates = sorted(
            (record for record in records if record["language"] == language),
            key=lambda record: (record["parallel_group_id"], record["prompt_id"]),
        )
        selected.append(candidates[0])
    return selected


def select_pilot_prompts(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    selected = list(records)
    random.Random(seed).shuffle(selected)
    return selected


def build_requests(records: list[dict[str, Any]], *, smoke: bool, seed: int) -> list[dict[str, Any]]:
    measured = select_smoke_prompts(records) if smoke else select_pilot_prompts(records, seed)
    return [
        {"prompt": select_smoke_prompts(records)[0], "warmup_or_measured": "warmup"},
        *({"prompt": record, "warmup_or_measured": "measured"} for record in measured),
    ]


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def chat_completion_url(base_url: str) -> str:
    return f"{normalize_base_url(base_url)}/v1/chat/completions"


def models_url(base_url: str) -> str:
    return f"{normalize_base_url(base_url)}/v1/models"


def encode_len(tokenizer: Any, text: str) -> int | None:
    if tokenizer is None:
        return None
    return len(tokenizer.encode(text, add_special_tokens=False))


def formatted_text_for_count(tokenizer: Any, record: dict[str, Any]) -> str:
    messages = prompt_messages(record)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
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
    return json.dumps(messages, ensure_ascii=False)


def parse_openai_stream_line(line: str) -> str | None:
    if not line.startswith("data: "):
        return None
    data = line[6:]
    if data == "[DONE]":
        return None
    obj = json.loads(data)
    choice = obj.get("choices", [{}])[0]
    delta = choice.get("delta") or {}
    if "content" in delta:
        return delta["content"] or ""
    text = choice.get("text")
    return text or ""


def request_streaming_completion(
    *,
    requests_module: Any,
    base_url: str,
    model_id: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    start = time.perf_counter()
    first: float | None = None
    chunks: list[str] = []
    response = requests_module.post(
        chat_completion_url(base_url),
        json=payload,
        stream=True,
        timeout=timeout_seconds,
    )
    with response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            chunk = parse_openai_stream_line(raw_line)
            if chunk is None:
                continue
            if chunk and first is None:
                first = time.perf_counter()
            chunks.append(chunk)
    end = time.perf_counter()
    generated_text = "".join(chunks)
    if not generated_text.strip():
        raise RuntimeError("SGLang stream completed but produced empty text")
    return {
        "ttft_ms": None if first is None else (first - start) * 1000,
        "total_latency_ms": (end - start) * 1000,
        "generated_text": generated_text,
    }


def make_raw_record(
    *,
    experiment_id: str,
    model_label: str,
    model_id: str,
    base_url: str,
    prompt_record: dict[str, Any],
    raw_user_prompt_tokens: int | None,
    formatted_input_tokens: int | None,
    output_tokens: int | None,
    max_tokens: int,
    temperature: float,
    seed: int,
    ttft_ms: float | None,
    total_latency_ms: float | None,
    generated_text_preview: str | None,
    success: bool,
    error_type: str | None,
    error_message: str | None,
    warmup_or_measured: str,
) -> dict[str, Any]:
    tok_s = None
    if output_tokens is not None and total_latency_ms is not None and total_latency_ms > 0:
        tok_s = output_tokens / (total_latency_ms / 1000)
    return {
        "experiment_id": experiment_id,
        "experiment_type": EXPERIMENT_TYPE,
        "timestamp_utc": utc_now_iso(),
        "model_id": model_id,
        "model_label": model_label,
        "runtime": RUNTIME,
        "base_url": normalize_base_url(base_url),
        "prompt_id": prompt_record["prompt_id"],
        "language": prompt_record["language"],
        "raw_user_prompt_text_hash": sha256_text(prompt_record["user_prompt"]),
        "raw_user_prompt_tokens": raw_user_prompt_tokens,
        "formatted_input_tokens": formatted_input_tokens,
        "output_tokens": output_tokens,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "ttft_ms": round_or_none(ttft_ms, 3),
        "total_latency_ms": round_or_none(total_latency_ms, 3),
        "end_to_end_output_tokens_per_second": round_or_none(tok_s, 3),
        "generated_text_preview": generated_text_preview,
        "success": success,
        "error_type": error_type,
        "error_message": error_message,
        "warmup_or_measured": warmup_or_measured,
    }


def validate_raw_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in RAW_FIELDS:
        if field not in record:
            errors.append(f"missing field '{field}'")
    if errors:
        return errors
    if record["experiment_type"] != EXPERIMENT_TYPE:
        errors.append(f"experiment_type must be {EXPERIMENT_TYPE}")
    if record["model_label"] not in MODEL_CONFIGS:
        errors.append("model_label must be sarvam or qwen")
    elif record["model_id"] != MODEL_CONFIGS[record["model_label"]]["model_id"]:
        errors.append("model_id does not match model_label")
    if record["runtime"] != RUNTIME:
        errors.append(f"runtime must be {RUNTIME}")
    if record["language"] not in LANGUAGE_ORDER:
        errors.append("unsupported language")
    if record["warmup_or_measured"] not in {"warmup", "measured"}:
        errors.append("warmup_or_measured must be warmup or measured")
    if not isinstance(record["success"], bool):
        errors.append("success must be boolean")
    return errors


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
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    measured = [row for row in records if row["warmup_or_measured"] == "measured"]
    summary: list[dict[str, Any]] = []
    for language in LANGUAGE_ORDER:
        language_rows = [row for row in measured if row["language"] == language]
        successes = [row for row in language_rows if row["success"]]
        ttft = [float(row["ttft_ms"]) for row in successes if row["ttft_ms"] is not None]
        total = [
            float(row["total_latency_ms"])
            for row in successes
            if row["total_latency_ms"] is not None
        ]
        raw_tokens = [
            float(row["raw_user_prompt_tokens"])
            for row in successes
            if row["raw_user_prompt_tokens"] is not None
        ]
        formatted_tokens = [
            float(row["formatted_input_tokens"])
            for row in successes
            if row["formatted_input_tokens"] is not None
        ]
        output_tokens = [
            float(row["output_tokens"])
            for row in successes
            if row["output_tokens"] is not None
        ]
        tok_s = [
            float(row["end_to_end_output_tokens_per_second"])
            for row in successes
            if row["end_to_end_output_tokens_per_second"] is not None
        ]
        summary.append(
            {
                "language": language,
                "prompt_count": len(language_rows),
                "successful_request_count": len(successes),
                "failed_request_count": len(language_rows) - len(successes),
                "mean_raw_user_prompt_tokens": round_or_none(mean(raw_tokens), 3),
                "mean_formatted_input_tokens": round_or_none(mean(formatted_tokens), 3),
                "median_ttft_ms": round_or_none(median(ttft), 3),
                "p90_ttft_ms": round_or_none(quantile(ttft, 0.90), 3),
                "p90_note": "exploratory; n<20" if 0 < len(ttft) < 20 else "",
                "median_total_latency_ms": round_or_none(median(total), 3),
                "mean_output_tokens": round_or_none(mean(output_tokens), 3),
                "mean_end_to_end_output_tokens_per_second": round_or_none(mean(tok_s), 3),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in SUMMARY_FIELDS})


def render_summary_markdown(
    summary: list[dict[str, Any]],
    *,
    model_label: str = "sarvam",
    model_id: str = MODEL_ID,
) -> str:
    config = MODEL_CONFIGS[model_label]
    lines = [
        f"# {config['title']}",
        "",
        CLAIM_BOUNDARY,
        "",
        f"- Model: `{model_label}` (`{model_id}`)",
        "",
        "| Language | n | successes | median TTFT ms | p90 TTFT ms | median total ms | mean raw toks | mean formatted toks | mean output toks | mean tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {language} | {prompt_count} | {successful_request_count} | "
            "{median_ttft_ms} | {p90_ttft_ms} | {median_total_latency_ms} | "
            "{mean_raw_user_prompt_tokens} | {mean_formatted_input_tokens} | "
            "{mean_output_tokens} | {mean_end_to_end_output_tokens_per_second} |".format(
                **{key: ("None" if value is None else value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            f"Meeting wording: {config['meeting_label']}, single concurrency, frozen 48-prompt natural suite. Do not describe this as production performance.",
        ]
    )
    return "\n".join(lines) + "\n"


def load_tokenizer(model_label: str, local_files_only: bool) -> Any:
    from transformers import AutoTokenizer  # type: ignore

    config = MODEL_CONFIGS[model_label]
    return AutoTokenizer.from_pretrained(
        config["model_id"],
        trust_remote_code=bool(config["trust_remote_code"]),
        local_files_only=local_files_only,
    )


def health_check(requests_module: Any, base_url: str, timeout_seconds: float) -> None:
    response = requests_module.get(models_url(base_url), timeout=timeout_seconds)
    response.raise_for_status()


def run_client(args: argparse.Namespace) -> int:
    import requests  # type: ignore

    mode = "smoke" if args.smoke else "pilot"
    model_label = args.model
    model_id = MODEL_CONFIGS[model_label]["model_id"]
    experiment_id = args.experiment_id or default_experiment_id(model_label, mode)
    run_dir = args.output_root / experiment_id
    raw_path = run_dir / "raw_requests.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path.touch(exist_ok=True)

    metadata = {
        "experiment_id": experiment_id,
        "experiment_type": EXPERIMENT_TYPE,
        "runtime": RUNTIME,
        "model_id": model_id,
        "model_label": model_label,
        "base_url": normalize_base_url(args.base_url),
        "mode": mode,
        "command": " ".join(sys.argv),
        "start_timestamp_utc": utc_now_iso(),
        "end_timestamp_utc": None,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "limitations": CLAIM_BOUNDARY,
        "fatal_error": None,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    try:
        health_check(requests, args.base_url, args.timeout_seconds)
        tokenizer = load_tokenizer(model_label, args.local_files_only)
        prompts = load_natural_prompt_suite(args.prompts)
        requests_to_run = build_requests(prompts, smoke=args.smoke, seed=args.seed)

        for request in requests_to_run:
            prompt_record = request["prompt"]
            raw_tokens = encode_len(tokenizer, prompt_record["user_prompt"])
            formatted_tokens = encode_len(
                tokenizer, formatted_text_for_count(tokenizer, prompt_record)
            )
            try:
                measurement = request_streaming_completion(
                    requests_module=requests,
                    base_url=args.base_url,
                    model_id=model_id,
                    messages=prompt_messages(prompt_record),
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout_seconds=args.timeout_seconds,
                )
                generated_text = measurement["generated_text"]
                output_tokens = encode_len(tokenizer, generated_text)
                row = make_raw_record(
                    experiment_id=experiment_id,
                    model_label=model_label,
                    model_id=model_id,
                    base_url=args.base_url,
                    prompt_record=prompt_record,
                    raw_user_prompt_tokens=raw_tokens,
                    formatted_input_tokens=formatted_tokens,
                    output_tokens=output_tokens,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    seed=args.seed,
                    ttft_ms=measurement["ttft_ms"],
                    total_latency_ms=measurement["total_latency_ms"],
                    generated_text_preview=" ".join(generated_text.split())[:200],
                    success=True,
                    error_type=None,
                    error_message=None,
                    warmup_or_measured=request["warmup_or_measured"],
                )
            except Exception as exc:  # noqa: BLE001 - preserve benchmark failure
                row = make_raw_record(
                    experiment_id=experiment_id,
                    model_label=model_label,
                    model_id=model_id,
                    base_url=args.base_url,
                    prompt_record=prompt_record,
                    raw_user_prompt_tokens=raw_tokens,
                    formatted_input_tokens=formatted_tokens,
                    output_tokens=None,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    seed=args.seed,
                    ttft_ms=None,
                    total_latency_ms=None,
                    generated_text_preview=None,
                    success=False,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    warmup_or_measured=request["warmup_or_measured"],
                )
            append_jsonl(raw_path, row)
            print(
                row["warmup_or_measured"],
                row["prompt_id"],
                row["language"],
                "ok" if row["success"] else row["error_type"],
                "ttft",
                row["ttft_ms"],
                flush=True,
            )
            if not row["success"]:
                break
    except Exception as exc:  # noqa: BLE001
        metadata["fatal_error"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        rows = read_jsonl(raw_path)
        summary = summarize_records(rows)
        write_csv(run_dir / "summary_by_language.csv", summary)
        (run_dir / "human_readable_summary.md").write_text(
            render_summary_markdown(summary, model_label=model_label, model_id=model_id),
            encoding="utf-8",
        )
        metadata["end_timestamp_utc"] = utc_now_iso()
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if metadata["fatal_error"] is not None:
        return 1
    expected = 4 if args.smoke else 48
    successes = [
        row for row in read_jsonl(raw_path)
        if row["warmup_or_measured"] == "measured" and row["success"]
    ]
    print(f"DONE: {run_dir}", flush=True)
    return 0 if len(successes) == expected else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an H100 SGLang OpenAI-compatible streaming pilot."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="sarvam")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    return run_client(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
