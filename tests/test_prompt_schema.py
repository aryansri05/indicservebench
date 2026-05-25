from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from indicservebench.prompt_schema import (  # noqa: E402
    LANGUAGE_TO_SCRIPT,
    REQUIRED_FIELDS,
    load_prompt_jsonl,
    validate_prompt_records,
)


PROMPTS_PATH = PROJECT_ROOT / "prompts" / "prototype_prompts.jsonl"


def test_prototype_prompts_are_valid() -> None:
    records = load_prompt_jsonl(PROMPTS_PATH)
    assert validate_prompt_records(records) == []


def test_prototype_prompt_scope_is_exactly_12_groups_and_36_records() -> None:
    records = load_prompt_jsonl(PROMPTS_PATH)
    groups = {record["parallel_group_id"] for record in records}
    assert len(records) == 36
    assert len(groups) == 12


def test_each_group_has_hindi_tamil_and_hinglish() -> None:
    records = load_prompt_jsonl(PROMPTS_PATH)
    expected_languages = set(LANGUAGE_TO_SCRIPT)
    by_group: dict[str, set[str]] = {}

    for record in records:
        by_group.setdefault(record["parallel_group_id"], set()).add(record["language"])

    assert by_group
    for languages in by_group.values():
        assert languages == expected_languages


def test_required_fields_are_present() -> None:
    records = load_prompt_jsonl(PROMPTS_PATH)
    for record in records:
        assert set(REQUIRED_FIELDS).issubset(record)


def test_prototype_prompts_are_natural_only() -> None:
    records = load_prompt_jsonl(PROMPTS_PATH)
    assert {record["suite_type"] for record in records} == {"natural"}
    assert {record["token_control_method"] for record in records} == {"not_applicable"}


def test_scripts_match_language_groups() -> None:
    records = load_prompt_jsonl(PROMPTS_PATH)
    for record in records:
        assert record["script"] == LANGUAGE_TO_SCRIPT[record["language"]]
