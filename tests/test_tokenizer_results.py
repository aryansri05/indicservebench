from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from indicservebench.tokenizer_analysis import (  # noqa: E402
    RAW_RESULT_FIELDS,
    SUMMARY_FIELDS,
    build_row,
    prepare_experiment_paths,
    summarize_rows,
    token_count_from_tokenized,
    write_jsonl,
    write_metadata,
    write_summary_csv,
)


def sample_prompt_record() -> dict[str, object]:
    return {
        "prompt_id": "sample_hi",
        "parallel_group_id": "sample",
        "language": "hi",
        "script": "Devanagari",
        "suite_type": "natural",
        "workload_type": "short_128",
        "intent_category": "diagnostic",
        "system_prompt": "आप सहायक हैं।",
        "context_text": "",
        "user_prompt": "कृपया मदद करें।",
    }


def test_raw_result_schema_and_computed_fields() -> None:
    row = build_row(
        experiment_id="unit_exp",
        generated_at_utc="2026-05-25T00:00:00+00:00",
        model_id="example/model",
        record=sample_prompt_record(),
        raw_user_prompt_tokens=4,
        formatted_input_tokens=11,
        tokenizer_load_success=True,
        raw_tokenization_success=True,
        template_success=True,
        chat_template_present=True,
        error_type=None,
        error_message=None,
        raw_error_message=None,
        template_error_message=None,
    )

    assert tuple(row.keys()) == RAW_RESULT_FIELDS
    assert row["character_count"] == len("कृपया मदद करें।")
    assert row["template_overhead_tokens"] == 7
    assert row["raw_tokens_per_character"] == 4 / len("कृपया मदद करें।")
    assert row["formatted_tokens_per_character"] == 11 / len("कृपया मदद करें।")
    assert row["tokenizer_load_success"] is True
    assert row["template_success"] is True


def test_summary_schema_integrity() -> None:
    row = build_row(
        experiment_id="unit_exp",
        generated_at_utc="2026-05-25T00:00:00+00:00",
        model_id="example/model",
        record=sample_prompt_record(),
        raw_user_prompt_tokens=4,
        formatted_input_tokens=11,
        tokenizer_load_success=True,
        raw_tokenization_success=True,
        template_success=True,
        chat_template_present=True,
        error_type=None,
        error_message=None,
        raw_error_message=None,
        template_error_message=None,
    )

    summary = summarize_rows([row])
    assert len(summary) == 1
    assert tuple(summary[0].keys()) == SUMMARY_FIELDS
    assert summary[0]["prompt_count"] == 1
    assert summary[0]["mean_raw_user_prompt_tokens"] == 4
    assert summary[0]["mean_template_overhead_tokens"] == 7
    assert summary[0]["mean_formatted_input_tokens"] == 11
    assert summary[0]["failures"] == 0


def test_experiment_output_directory_isolated(tmp_path: Path) -> None:
    paths = prepare_experiment_paths(tmp_path, "unit_exp")
    row = build_row(
        experiment_id="unit_exp",
        generated_at_utc="2026-05-25T00:00:00+00:00",
        model_id="example/model",
        record=sample_prompt_record(),
        raw_user_prompt_tokens=4,
        formatted_input_tokens=11,
        tokenizer_load_success=True,
        raw_tokenization_success=True,
        template_success=True,
        chat_template_present=True,
        error_type=None,
        error_message=None,
        raw_error_message=None,
        template_error_message=None,
    )

    write_jsonl(paths.raw_jsonl, [row])
    write_summary_csv(paths.summary_csv, [row])
    write_metadata(
        paths.metadata_json,
        experiment_id="unit_exp",
        generated_at_utc="2026-05-25T00:00:00+00:00",
        prompts_path=Path("prompts/prototype_prompts.jsonl"),
        models_config_path=Path("configs/models.yaml"),
        model_configs=[{"model_id": "example/model"}],
        rows=[row],
        paths=paths,
    )

    assert paths.output_dir == tmp_path / "unit_exp"
    assert paths.raw_jsonl == tmp_path / "unit_exp" / "raw.jsonl"
    assert paths.summary_csv == tmp_path / "unit_exp" / "summary.csv"
    assert paths.metadata_json == tmp_path / "unit_exp" / "metadata.json"
    assert not (tmp_path / "raw.jsonl").exists()
    assert not (tmp_path / "summary.csv").exists()
    assert not (tmp_path / "metadata.json").exists()

    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert metadata["gpu_or_serving_code_used"] is False
    assert "Actual serving latency will be measured later on GPU." in metadata[
        "diagnostic_note"
    ]

    with pytest.raises(FileExistsError):
        prepare_experiment_paths(tmp_path, "unit_exp")


def test_token_count_accepts_batch_encoding_like_objects() -> None:
    class FakeBatchEncoding:
        def get(self, key: str) -> list[int]:
            assert key == "input_ids"
            return [101, 202, 303]

    count, error = token_count_from_tokenized(FakeBatchEncoding())
    assert count == 3
    assert error is None
