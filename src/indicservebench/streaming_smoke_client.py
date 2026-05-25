"""Runtime-agnostic OpenAI-compatible streaming smoke client.

This module is for a future tightly scoped H100 smoke test. It does not install
or launch any serving runtime and contains no GPU-specific commands.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Iterator


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
LANGUAGE_ORDER = ("hi", "ta", "hinglish")
SMOKE_MAX_TOKENS = 32

RESULT_FIELDS = (
    "run_id",
    "model_id",
    "runtime_label",
    "prompt_id",
    "language",
    "request_start_utc",
    "input_prompt_text_hash",
    "first_content_token_utc",
    "completion_utc",
    "ttft_ms",
    "total_latency_ms",
    "streamed_text_received",
    "success",
    "error_message",
)


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def default_run_id() -> str:
    return "streaming_smoke_" + utc_now_iso().replace("-", "").replace(":", "").replace(
        "+", "Z"
    )


def chat_completions_url(server_base_url: str) -> str:
    base = server_base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def select_smoke_prompts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select exactly one existing short natural prompt per language."""

    selected: list[dict[str, Any]] = []
    for language in LANGUAGE_ORDER:
        candidates = sorted(
            (
                record
                for record in records
                if record["language"] == language
                and record["suite_type"] == "natural"
                and record["workload_type"] == "short_128"
            ),
            key=lambda record: record["prompt_id"],
        )
        if not candidates:
            raise ValueError(f"No short natural prompt found for language '{language}'")
        selected.append(candidates[0])

    if len(selected) != 3:
        raise ValueError("Smoke test must select exactly three prompts")
    return selected


def input_prompt_text_hash(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_chat_completion_payload(model_id: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": prompt_messages(record),
        "stream": True,
        "max_tokens": SMOKE_MAX_TOKENS,
        "temperature": 0,
    }


def parse_sse_events(lines: Iterable[bytes | str]) -> Iterator[str]:
    """Yield SSE data payloads from an iterable of response lines."""

    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.rstrip("\r\n")

        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue

        if line.startswith(":"):
            continue

        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())

    if data_lines:
        yield "\n".join(data_lines)


def extract_content_from_stream_chunk(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices")
    if not choices:
        return ""

    first_choice = choices[0] or {}
    delta = first_choice.get("delta") or {}
    content = delta.get("content")
    if content is None:
        return ""
    return str(content)


def iter_streamed_content(lines: Iterable[bytes | str]) -> Iterator[str]:
    """Yield only non-empty content deltas from OpenAI-compatible SSE lines."""

    for event_data in parse_sse_events(lines):
        if event_data == "[DONE]":
            break
        try:
            chunk = json.loads(event_data)
        except json.JSONDecodeError:
            continue

        content = extract_content_from_stream_chunk(chunk)
        if content:
            yield content


def validate_result_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in RESULT_FIELDS:
        if field not in record:
            errors.append(f"missing field '{field}'")

    if errors:
        return errors

    string_fields = (
        "run_id",
        "model_id",
        "runtime_label",
        "prompt_id",
        "language",
        "request_start_utc",
        "input_prompt_text_hash",
        "completion_utc",
        "streamed_text_received",
    )
    for field in string_fields:
        if not isinstance(record[field], str):
            errors.append(f"field '{field}' must be a string")

    if record["first_content_token_utc"] is not None and not isinstance(
        record["first_content_token_utc"], str
    ):
        errors.append("field 'first_content_token_utc' must be a string or null")

    for field in ("ttft_ms", "total_latency_ms"):
        if record[field] is not None and not isinstance(record[field], (int, float)):
            errors.append(f"field '{field}' must be numeric or null")

    if not isinstance(record["success"], bool):
        errors.append("field 'success' must be boolean")

    if record["error_message"] is not None and not isinstance(record["error_message"], str):
        errors.append("field 'error_message' must be a string or null")

    if record["language"] not in LANGUAGE_ORDER:
        errors.append(f"unsupported language '{record['language']}'")

    return errors


def append_result(path: Path, record: dict[str, Any]) -> None:
    errors = validate_result_record(record)
    if errors:
        raise ValueError("; ".join(errors))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def make_result_record(
    *,
    run_id: str,
    model_id: str,
    runtime_label: str,
    record: dict[str, Any],
    request_start_utc: str,
    prompt_hash: str,
    first_content_token_utc: str | None,
    completion_utc: str,
    ttft_ms: float | None,
    total_latency_ms: float | None,
    streamed_text_received: str,
    success: bool,
    error_message: str | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "model_id": model_id,
        "runtime_label": runtime_label,
        "prompt_id": record["prompt_id"],
        "language": record["language"],
        "request_start_utc": request_start_utc,
        "input_prompt_text_hash": prompt_hash,
        "first_content_token_utc": first_content_token_utc,
        "completion_utc": completion_utc,
        "ttft_ms": ttft_ms,
        "total_latency_ms": total_latency_ms,
        "streamed_text_received": streamed_text_received,
        "success": success,
        "error_message": error_message,
    }


def send_streaming_request(
    *,
    server_base_url: str,
    model_id: str,
    runtime_label: str,
    run_id: str,
    record: dict[str, Any],
    timeout_seconds: float,
    api_key: str | None,
) -> dict[str, Any]:
    payload = build_chat_completion_payload(model_id, record)
    payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request_start_utc = utc_now_iso()
    request_start_ns = time.perf_counter_ns()
    messages = payload["messages"]
    prompt_hash = input_prompt_text_hash(messages)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        chat_completions_url(server_base_url),
        data=payload_bytes,
        headers=headers,
        method="POST",
    )

    first_content_token_utc: str | None = None
    first_content_token_ns: int | None = None
    streamed_text_parts: list[str] = []
    success = False
    error_message: str | None = None

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            for content in iter_streamed_content(response):
                if first_content_token_ns is None:
                    first_content_token_ns = time.perf_counter_ns()
                    first_content_token_utc = utc_now_iso()
                streamed_text_parts.append(content)

        if streamed_text_parts:
            success = True
        else:
            error_message = "stream completed without any content tokens"
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        error_message = f"HTTPError {exc.code}: {body or exc.reason}"
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"

    completion_ns = time.perf_counter_ns()
    completion_utc = utc_now_iso()
    ttft_ms = None
    if first_content_token_ns is not None:
        ttft_ms = (first_content_token_ns - request_start_ns) / 1_000_000

    total_latency_ms = (completion_ns - request_start_ns) / 1_000_000

    return make_result_record(
        run_id=run_id,
        model_id=model_id,
        runtime_label=runtime_label,
        record=record,
        request_start_utc=request_start_utc,
        prompt_hash=prompt_hash,
        first_content_token_utc=first_content_token_utc,
        completion_utc=completion_utc,
        ttft_ms=ttft_ms,
        total_latency_ms=total_latency_ms,
        streamed_text_received="".join(streamed_text_parts),
        success=success,
        error_message=error_message,
    )


def write_metadata(
    path: Path,
    *,
    run_id: str,
    server_base_url: str,
    model_id: str,
    runtime_label: str,
    prompt_ids: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_id": run_id,
        "created_at_utc": utc_now_iso(),
        "server_base_url": server_base_url,
        "model_id": model_id,
        "runtime_label": runtime_label,
        "prompt_ids": prompt_ids,
        "max_tokens": SMOKE_MAX_TOKENS,
        "concurrency": 1,
        "result_schema": list(RESULT_FIELDS),
        "gpu_or_runtime_launch_performed_by_client": False,
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a three-prompt OpenAI-compatible streaming smoke test."
    )
    parser.add_argument("--server-base-url", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--runtime-label", required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=PROJECT_ROOT / "prompts" / "prototype_prompts.jsonl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "streaming_smoke",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="Optional bearer token. Defaults to OPENAI_API_KEY when set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or default_run_id()
    output_dir = args.output_root / run_id
    raw_jsonl = output_dir / "raw.jsonl"
    metadata_json = output_dir / "metadata.json"

    records = load_prompt_jsonl(args.prompts)
    assert_valid_prompt_records(records)
    selected_prompts = select_smoke_prompts(records)

    write_metadata(
        metadata_json,
        run_id=run_id,
        server_base_url=args.server_base_url,
        model_id=args.model_id,
        runtime_label=args.runtime_label,
        prompt_ids=[record["prompt_id"] for record in selected_prompts],
    )

    for record in selected_prompts:
        result = send_streaming_request(
            server_base_url=args.server_base_url,
            model_id=args.model_id,
            runtime_label=args.runtime_label,
            run_id=run_id,
            record=record,
            timeout_seconds=args.timeout_seconds,
            api_key=args.api_key,
        )
        append_result(raw_jsonl, result)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "output_dir": str(output_dir),
                "raw_jsonl": str(raw_jsonl),
                "metadata_json": str(metadata_json),
                "prompt_count": len(selected_prompts),
                "max_tokens": SMOKE_MAX_TOKENS,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
