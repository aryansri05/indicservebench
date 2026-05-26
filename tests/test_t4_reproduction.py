from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from indicservebench.t4_reproduction_analysis import (  # noqa: E402
    aggregate_records,
    quantile,
    repeated_prompt_ids_in_slowest,
    select_slowest_requests,
)
from indicservebench.t4_reproduction_runner import (  # noqa: E402
    BACKEND,
    LANGUAGE_ORDER,
    MODEL_ID,
    RAW_REQUEST_FIELDS,
    T4EnvironmentError,
    build_measured_requests,
    build_warmup_requests,
    load_recovered_prompts,
    make_raw_record,
    validate_raw_record,
    validate_t4_environment,
)


PROMPTS_PATH = PROJECT_ROOT / "prompts" / "sarvam2b_t4_original_prompts.jsonl"


def test_recovered_old_prompt_dataset_is_present_and_valid() -> None:
    records = load_recovered_prompts(PROMPTS_PATH)

    assert len(records) == 30
    assert Counter(record["language"] for record in records) == {
        "hindi": 10,
        "tamil": 10,
        "codemix": 10,
    }
    assert records[0]["prompt_id"] == "sarvam2b_t4_hindi_001"
    assert records[0]["text"] == "भारत की राजधानी क्या है?"
    assert records[10]["prompt_id"] == "sarvam2b_t4_tamil_001"
    assert records[20]["prompt_id"] == "sarvam2b_t4_codemix_001"


def test_measured_request_order_is_deterministic_shuffled_and_balanced() -> None:
    records = load_recovered_prompts(PROMPTS_PATH)
    first = build_measured_requests(records, repetitions=5, seed=42, smoke_test=False)
    second = build_measured_requests(records, repetitions=5, seed=42, smoke_test=False)

    first_ids = [request["prompt"]["prompt_id"] for request in first]
    second_ids = [request["prompt"]["prompt_id"] for request in second]
    assert first_ids == second_ids
    assert Counter(request["prompt"]["language"] for request in first) == {
        "hindi": 50,
        "tamil": 50,
        "codemix": 50,
    }

    unshuffled_ids = [
        record["prompt_id"]
        for _repetition in range(5)
        for record in records
    ]
    assert first_ids != unshuffled_ids
    assert len({request["prompt"]["language"] for request in first[:10]}) > 1


def test_warmups_and_smoke_mode_use_one_prompt_per_language() -> None:
    records = load_recovered_prompts(PROMPTS_PATH)

    warmups = build_warmup_requests(records)
    smoke_measured = build_measured_requests(
        records, repetitions=5, seed=42, smoke_test=True
    )

    assert [request["warmup_or_measured"] for request in warmups] == [
        "warmup",
        "warmup",
        "warmup",
    ]
    assert [request["prompt"]["language"] for request in warmups] == list(LANGUAGE_ORDER)
    assert Counter(request["prompt"]["language"] for request in smoke_measured) == {
        "hindi": 1,
        "tamil": 1,
        "codemix": 1,
    }
    assert {request["warmup_or_measured"] for request in smoke_measured} == {"measured"}


def sample_raw_record(
    *,
    language: str = "hindi",
    prompt_id: str = "sarvam2b_t4_hindi_001",
    repetition_id: int = 1,
    warmup_or_measured: str = "measured",
    latency_ms: float = 100.0,
    input_tokens: int = 12,
    output_tokens: int = 64,
    success: bool = True,
) -> dict[str, object]:
    prompt_record = {
        "prompt_id": prompt_id,
        "language": language,
    }
    return make_raw_record(
        experiment_id="unit_exp",
        gpu_name="Tesla T4",
        gpu_environment="t4_environment",
        cuda_version="12.1",
        torch_version="2.5.0",
        transformers_version="4.51.0",
        dtype="float16",
        generation_config={"max_new_tokens": 64},
        random_seed=42,
        prompt_record=prompt_record,
        repetition_id=repetition_id,
        warmup_or_measured=warmup_or_measured,
        input_text_hash="a" * 64,
        input_tokens=input_tokens,
        output_tokens=output_tokens if success else None,
        generated_text_hash="b" * 64 if success else None,
        total_latency_ms=latency_ms if success else None,
        output_tokens_per_second=(output_tokens / (latency_ms / 1000)) if success else None,
        peak_gpu_memory_mb=1234.5 if success else None,
        success=success,
        error_type=None if success else "RuntimeError",
        error_message=None if success else "boom",
    )


def test_raw_request_schema_validation() -> None:
    record = sample_raw_record()

    assert tuple(record.keys()) == RAW_REQUEST_FIELDS
    assert record["model_id"] == MODEL_ID
    assert record["backend"] == BACKEND
    assert validate_raw_record(record) == []


def test_aggregation_excludes_warmups_and_counts_failures() -> None:
    records = [
        sample_raw_record(
            language="hindi",
            warmup_or_measured="warmup",
            latency_ms=9999,
            input_tokens=999,
            output_tokens=999,
        ),
        sample_raw_record(language="hindi", repetition_id=1, latency_ms=100, input_tokens=10),
        sample_raw_record(language="hindi", repetition_id=2, latency_ms=200, input_tokens=12),
        sample_raw_record(language="hindi", repetition_id=3, latency_ms=300, input_tokens=14),
        sample_raw_record(
            language="hindi",
            repetition_id=4,
            success=False,
            latency_ms=0,
            input_tokens=0,
            output_tokens=0,
        ),
    ]

    rows = aggregate_records(records)
    hindi = next(row for row in rows if row["language"] == "hindi")

    assert hindi["measured_request_count"] == 4
    assert hindi["successful_request_count"] == 3
    assert hindi["failed_request_count"] == 1
    assert hindi["mean_input_tokens"] == 12
    assert hindi["median_latency_ms"] == 200
    assert hindi["max_latency_ms"] == 300
    assert hindi["failure_rate"] == 0.25


def test_quantile_uses_linear_interpolation_not_max_as_p95() -> None:
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    assert quantile(values, 0.90) == pytest.approx(9.1)
    assert quantile(values, 0.95) == pytest.approx(9.55)


def test_selects_five_slowest_requests_per_language() -> None:
    records: list[dict[str, object]] = []
    for language in LANGUAGE_ORDER:
        for index in range(1, 7):
            records.append(
                sample_raw_record(
                    language=language,
                    prompt_id=f"{language}_{index}",
                    repetition_id=index,
                    latency_ms=float(index * 100),
                )
            )

    slowest = select_slowest_requests(records, per_language=5)

    assert Counter(row["language"] for row in slowest) == {
        "hindi": 5,
        "tamil": 5,
        "codemix": 5,
    }
    tamil_rows = [row for row in slowest if row["language"] == "tamil"]
    assert [row["total_latency_ms"] for row in tamil_rows] == [
        600.0,
        500.0,
        400.0,
        300.0,
        200.0,
    ]


def test_detects_repeated_tamil_prompt_ids_among_slowest() -> None:
    slowest_rows = [
        {"language": "tamil", "prompt_id": "ta_001"},
        {"language": "tamil", "prompt_id": "ta_001"},
        {"language": "tamil", "prompt_id": "ta_002"},
        {"language": "hindi", "prompt_id": "hi_001"},
    ]

    assert repeated_prompt_ids_in_slowest(slowest_rows, "tamil") == {"ta_001": 2}


def test_t4_environment_validation_requires_t4_or_explicit_override() -> None:
    assert validate_t4_environment("NVIDIA Tesla T4", allow_non_t4=False) == "t4_environment"
    assert validate_t4_environment("NVIDIA A100", allow_non_t4=True) == "non_t4_environment"

    with pytest.raises(T4EnvironmentError):
        validate_t4_environment("NVIDIA A100", allow_non_t4=False)

    with pytest.raises(T4EnvironmentError):
        validate_t4_environment(None, allow_non_t4=False)
