from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from indicservebench.h100_sglang_sarvam_pilot import (  # noqa: E402
    CLAIM_BOUNDARY,
    EXPERIMENT_TYPE,
    LANGUAGE_ORDER,
    MODEL_ID,
    RUNTIME,
    build_requests,
    make_raw_record,
    parse_openai_stream_line,
    render_summary_markdown,
    select_pilot_prompts,
    select_smoke_prompts,
    summarize_records,
    validate_raw_record,
)
from indicservebench.h100_transformers_streaming_pilot import load_natural_prompt_suite  # noqa: E402


PROMPTS_PATH = PROJECT_ROOT / "prompts" / "prototype_prompts.jsonl"


def sample_record(
    *,
    language: str = "hi",
    prompt_id: str = "nat_001_hi",
    warmup_or_measured: str = "measured",
    success: bool = True,
    ttft_ms: float = 100.0,
) -> dict[str, object]:
    return make_raw_record(
        experiment_id="unit_exp",
        base_url="http://127.0.0.1:30000/",
        prompt_record={"prompt_id": prompt_id, "language": language, "user_prompt": "hello"},
        raw_user_prompt_tokens=10 if success else None,
        formatted_input_tokens=30 if success else None,
        output_tokens=16 if success else None,
        max_tokens=32,
        temperature=0.0,
        seed=42,
        ttft_ms=ttft_ms if success else None,
        total_latency_ms=500.0 if success else None,
        generated_text_preview="ok" if success else None,
        success=success,
        error_type=None if success else "RuntimeError",
        error_message=None if success else "boom",
        warmup_or_measured=warmup_or_measured,
    )


def test_smoke_selects_one_prompt_per_language() -> None:
    records = load_natural_prompt_suite(PROMPTS_PATH)
    selected = select_smoke_prompts(records)

    assert [record["language"] for record in selected] == list(LANGUAGE_ORDER)
    assert [record["prompt_id"] for record in selected] == [
        "nat_001_en",
        "nat_001_hi",
        "nat_001_ta",
        "nat_001_hinglish",
    ]


def test_pilot_selection_is_deterministic_and_balanced() -> None:
    records = load_natural_prompt_suite(PROMPTS_PATH)
    first = select_pilot_prompts(records, seed=42)
    second = select_pilot_prompts(records, seed=42)

    assert [row["prompt_id"] for row in first] == [row["prompt_id"] for row in second]
    assert [row["prompt_id"] for row in first] != [row["prompt_id"] for row in records]
    assert Counter(row["language"] for row in first) == {
        "en": 12,
        "hi": 12,
        "ta": 12,
        "hinglish": 12,
    }


def test_build_requests_adds_single_warmup() -> None:
    records = load_natural_prompt_suite(PROMPTS_PATH)
    requests = build_requests(records, smoke=True, seed=42)

    assert requests[0]["warmup_or_measured"] == "warmup"
    assert Counter(request["warmup_or_measured"] for request in requests) == {
        "warmup": 1,
        "measured": 4,
    }


def test_raw_schema() -> None:
    row = sample_record()

    assert row["experiment_type"] == EXPERIMENT_TYPE
    assert row["model_id"] == MODEL_ID
    assert row["runtime"] == RUNTIME
    assert validate_raw_record(row) == []


def test_summary_excludes_warmup_and_counts_failures() -> None:
    rows = [
        sample_record(language="hi", warmup_or_measured="warmup", ttft_ms=9999),
        sample_record(language="hi", ttft_ms=100),
        sample_record(language="hi", ttft_ms=200),
        sample_record(language="hi", success=False),
    ]

    summary = summarize_records(rows)
    hi = next(row for row in summary if row["language"] == "hi")

    assert hi["prompt_count"] == 3
    assert hi["successful_request_count"] == 2
    assert hi["failed_request_count"] == 1
    assert hi["median_ttft_ms"] == 150


def test_parses_openai_streaming_delta_content() -> None:
    line = 'data: {"choices":[{"delta":{"content":"hello"}}]}'

    assert parse_openai_stream_line(line) == "hello"
    assert parse_openai_stream_line("data: [DONE]") is None


def test_markdown_claim_boundary() -> None:
    text = render_summary_markdown(summarize_records([sample_record(language="en", prompt_id="nat_001_en")]))

    assert CLAIM_BOUNDARY in text
    assert "not a production benchmark" in text
    assert "single concurrency" in text
