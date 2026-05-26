from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from indicservebench.h100_transformers_streaming_pilot import (  # noqa: E402
    EXPERIMENT_TYPE,
    LANGUAGE_ORDER,
    MODEL_CONFIGS,
    NON_H100_EXPERIMENT_LABEL,
    RAW_REQUEST_FIELDS,
    H100EnvironmentError,
    build_comparison_rows,
    build_requests,
    load_natural_prompt_suite,
    make_raw_record,
    quantile,
    render_comparison_summary,
    render_human_summary,
    select_pilot_prompts,
    select_smoke_prompts,
    summarize_records,
    validate_h100_environment,
    validate_raw_record,
)


PROMPTS_PATH = PROJECT_ROOT / "prompts" / "prototype_prompts.jsonl"


def sample_raw_record(
    *,
    model_label: str = "sarvam",
    language: str = "hi",
    prompt_id: str = "nat_001_hi",
    warmup_or_measured: str = "measured",
    success: bool = True,
    raw_tokens: int = 18,
    formatted_tokens: int = 60,
    output_tokens: int = 32,
    ttft_ms: float = 100.0,
    total_latency_ms: float = 500.0,
) -> dict[str, object]:
    return make_raw_record(
        experiment_id="unit_exp",
        experiment_label="h100_sxm_transformers_pilot",
        model_id=MODEL_CONFIGS[model_label]["model_id"],
        model_label=model_label,
        gpu_name="NVIDIA H100 80GB HBM3",
        gpu_vram_gb=79.18,
        torch_version="2.12.0+cu130",
        torch_cuda_version="13.0",
        transformers_version="4.51.3",
        prompt_record={"prompt_id": prompt_id, "language": language},
        raw_user_prompt_text_hash="a" * 64,
        raw_user_prompt_tokens=raw_tokens if success else None,
        formatted_input_tokens=formatted_tokens if success else None,
        output_tokens=output_tokens if success else None,
        max_new_tokens=32,
        do_sample=False,
        seed=42,
        ttft_ms=ttft_ms if success else None,
        total_latency_ms=total_latency_ms if success else None,
        end_to_end_output_tokens_per_second=(
            output_tokens / (total_latency_ms / 1000) if success else None
        ),
        generated_text_preview="short answer" if success else None,
        success=success,
        error_type=None if success else "RuntimeError",
        error_message=None if success else "boom",
        warmup_or_measured=warmup_or_measured,
    )


def test_h100_validation_requires_h100_or_explicit_override() -> None:
    assert (
        validate_h100_environment("NVIDIA H100 80GB HBM3", allow_non_h100=False)
        == "h100_sxm_transformers_pilot"
    )
    assert (
        validate_h100_environment("NVIDIA A100-SXM4-80GB", allow_non_h100=True)
        == NON_H100_EXPERIMENT_LABEL
    )

    with pytest.raises(H100EnvironmentError):
        validate_h100_environment("NVIDIA A100-SXM4-80GB", allow_non_h100=False)

    with pytest.raises(H100EnvironmentError):
        validate_h100_environment(None, allow_non_h100=False)


def test_natural_prompt_suite_is_loaded_and_validated() -> None:
    records = load_natural_prompt_suite(PROMPTS_PATH)

    assert len(records) == 48
    assert Counter(record["language"] for record in records) == {
        "en": 12,
        "hi": 12,
        "ta": 12,
        "hinglish": 12,
    }
    assert {record["suite_type"] for record in records} == {"natural"}


def test_smoke_prompt_selection_uses_one_prompt_per_language() -> None:
    records = load_natural_prompt_suite(PROMPTS_PATH)
    smoke = select_smoke_prompts(records)

    assert [record["language"] for record in smoke] == list(LANGUAGE_ORDER)
    assert [record["prompt_id"] for record in smoke] == [
        "nat_001_en",
        "nat_001_hi",
        "nat_001_ta",
        "nat_001_hinglish",
    ]

    requests = build_requests(records, smoke=True, seed=42)
    assert requests[0]["warmup_or_measured"] == "warmup"
    measured = [request for request in requests if request["warmup_or_measured"] == "measured"]
    assert Counter(request["prompt"]["language"] for request in measured) == {
        "en": 1,
        "hi": 1,
        "ta": 1,
        "hinglish": 1,
    }


def test_pilot_prompt_order_is_deterministic_and_shuffled() -> None:
    records = load_natural_prompt_suite(PROMPTS_PATH)
    first = select_pilot_prompts(records, seed=42)
    second = select_pilot_prompts(records, seed=42)

    assert [record["prompt_id"] for record in first] == [
        record["prompt_id"] for record in second
    ]
    assert [record["prompt_id"] for record in first] != [
        record["prompt_id"] for record in records
    ]
    assert Counter(record["language"] for record in first) == {
        "en": 12,
        "hi": 12,
        "ta": 12,
        "hinglish": 12,
    }


def test_raw_record_schema_validation() -> None:
    record = sample_raw_record()

    assert tuple(record.keys()) == RAW_REQUEST_FIELDS
    assert record["experiment_type"] == EXPERIMENT_TYPE
    assert validate_raw_record(record) == []


def test_summary_aggregation_excludes_warmups_and_counts_failures() -> None:
    records = [
        sample_raw_record(
            language="hi",
            warmup_or_measured="warmup",
            raw_tokens=999,
            formatted_tokens=999,
            ttft_ms=9999,
        ),
        sample_raw_record(language="hi", raw_tokens=10, formatted_tokens=50, ttft_ms=100),
        sample_raw_record(language="hi", raw_tokens=12, formatted_tokens=54, ttft_ms=200),
        sample_raw_record(language="hi", raw_tokens=14, formatted_tokens=58, ttft_ms=300),
        sample_raw_record(language="hi", success=False),
    ]

    rows = summarize_records(records)
    hi = next(row for row in rows if row["language"] == "hi")

    assert hi["prompt_count"] == 4
    assert hi["successful_request_count"] == 3
    assert hi["failed_request_count"] == 1
    assert hi["mean_raw_user_prompt_tokens"] == 12
    assert hi["mean_formatted_input_tokens"] == 54
    assert hi["median_ttft_ms"] == 200


def test_p90_uses_linear_interpolation() -> None:
    assert quantile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.90) == pytest.approx(9.1)


def test_comparison_aggregation_outputs_model_rows() -> None:
    sarvam_records = [
        sample_raw_record(
            model_label="sarvam",
            language="ta",
            prompt_id="nat_001_ta",
            raw_tokens=18,
            formatted_tokens=70,
            ttft_ms=120,
        )
    ]
    qwen_records = [
        sample_raw_record(
            model_label="qwen",
            language="ta",
            prompt_id="nat_001_ta",
            raw_tokens=81,
            formatted_tokens=140,
            ttft_ms=220,
        )
    ]

    rows = build_comparison_rows(sarvam_records, qwen_records)
    tamil_rows = [row for row in rows if row["language"] == "ta"]

    assert [row["model_label"] for row in tamil_rows] == ["sarvam", "qwen"]
    assert tamil_rows[0]["average_raw_user_prompt_tokens"] == 18
    assert tamil_rows[1]["median_ttft_ms"] == 220


def test_generated_markdown_preserves_claim_boundary() -> None:
    summary = render_human_summary(
        model_label="sarvam",
        model_id=MODEL_CONFIGS["sarvam"]["model_id"],
        mode="smoke",
        summary_rows=summarize_records([sample_raw_record(language="en", prompt_id="nat_001_en")]),
    )
    comparison = render_comparison_summary(
        build_comparison_rows(
            [sample_raw_record(model_label="sarvam", language="en", prompt_id="nat_001_en")],
            [sample_raw_record(model_label="qwen", language="en", prompt_id="nat_001_en")],
        )
    )

    for text in (summary, comparison):
        assert "preliminary" in text
        assert "not a production benchmark" in text
        assert "No CUDA/kernel bottleneck is claimed" in text
        assert "does not establish causality" in text or "does not prove" in text
