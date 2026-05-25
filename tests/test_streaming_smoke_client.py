from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from indicservebench.prompt_schema import load_prompt_jsonl  # noqa: E402
from indicservebench.streaming_smoke_client import (  # noqa: E402
    RESULT_FIELDS,
    iter_streamed_content,
    make_result_record,
    select_smoke_prompts,
    validate_result_record,
)


PROMPTS_PATH = PROJECT_ROOT / "prompts" / "prototype_prompts.jsonl"


def test_selects_exactly_one_short_natural_prompt_per_language() -> None:
    records = load_prompt_jsonl(PROMPTS_PATH)
    selected = select_smoke_prompts(records)

    assert len(selected) == 4
    assert [record["language"] for record in selected] == ["en", "hi", "ta", "hinglish"]
    assert {record["parallel_group_id"] for record in selected} == {"nat_001"}
    assert {record["suite_type"] for record in selected} == {"natural"}
    assert {record["workload_type"] for record in selected} == {"short_128"}


def test_result_schema_validation() -> None:
    record = {
        "prompt_id": "nat_001_hi",
        "language": "hi",
    }
    result = make_result_record(
        run_id="unit_run",
        model_id="sarvamai/sarvam-30b-fp8",
        runtime_label="sglang",
        record=record,
        request_start_utc="2026-05-25T00:00:00.000+00:00",
        prompt_hash="a" * 64,
        first_content_token_utc="2026-05-25T00:00:00.100+00:00",
        completion_utc="2026-05-25T00:00:00.500+00:00",
        ttft_ms=100.0,
        total_latency_ms=500.0,
        streamed_text_received="नमस्ते",
        success=True,
        error_message=None,
    )

    assert tuple(result.keys()) == RESULT_FIELDS
    assert validate_result_record(result) == []


def test_empty_non_content_chunk_before_first_content_token_is_ignored() -> None:
    empty_chunk = {"choices": [{"delta": {"role": "assistant"}}]}
    content_chunk = {"choices": [{"delta": {"content": "नमस्ते"}}]}
    done = "[DONE]"
    lines = [
        f"data: {json.dumps(empty_chunk)}\n",
        "\n",
        f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n",
        "\n",
        f"data: {done}\n",
        "\n",
    ]

    assert list(iter_streamed_content(lines)) == ["नमस्ते"]
