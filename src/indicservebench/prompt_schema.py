"""Prompt schema validation for IndicServeBench.

This module is intentionally CPU-only and has no model-serving dependency.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "schema_version",
    "prompt_id",
    "parallel_group_id",
    "language",
    "script",
    "suite_type",
    "workload_type",
    "intent_category",
    "system_prompt",
    "context_text",
    "user_prompt",
    "target_input_token_bucket",
    "expected_output_cap",
    "shared_prefix_id",
    "token_control_method",
    "notes",
)

LANGUAGE_TO_SCRIPT = {
    "hi": "Devanagari",
    "ta": "Tamil",
    "hinglish": "Latin",
}

ALLOWED_SUITE_TYPES = {"natural", "token_controlled"}
ALLOWED_WORKLOAD_TYPES = {"short_128", "context_1024", "long_3584"}


class PromptSchemaError(ValueError):
    """Raised when prompt records do not match the expected schema."""


def _location(record: dict[str, Any], record_index: int | None) -> str:
    prompt_id = record.get("prompt_id")
    if isinstance(prompt_id, str) and prompt_id:
        return prompt_id
    if record_index is None:
        return "<record>"
    return f"record[{record_index}]"


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_prompt_record(
    record: dict[str, Any], record_index: int | None = None
) -> list[str]:
    """Return validation errors for a single prompt record."""

    errors: list[str] = []
    loc = _location(record, record_index)

    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"{loc}: missing required field '{field}'")

    if errors:
        return errors

    for field in (
        "schema_version",
        "prompt_id",
        "parallel_group_id",
        "language",
        "script",
        "suite_type",
        "workload_type",
        "intent_category",
        "token_control_method",
        "notes",
    ):
        if not _is_non_empty_string(record[field]):
            errors.append(f"{loc}: field '{field}' must be a non-empty string")

    for field in ("system_prompt", "context_text", "user_prompt"):
        if not isinstance(record[field], str):
            errors.append(f"{loc}: field '{field}' must be a string")

    if not _is_non_empty_string(record["user_prompt"]):
        errors.append(f"{loc}: field 'user_prompt' must not be empty")

    language = record["language"]
    if language not in LANGUAGE_TO_SCRIPT:
        errors.append(f"{loc}: unsupported language '{language}'")
    elif record["script"] != LANGUAGE_TO_SCRIPT[language]:
        errors.append(
            f"{loc}: script '{record['script']}' does not match language '{language}'"
        )

    if record["suite_type"] not in ALLOWED_SUITE_TYPES:
        errors.append(f"{loc}: unsupported suite_type '{record['suite_type']}'")

    if record["workload_type"] not in ALLOWED_WORKLOAD_TYPES:
        errors.append(f"{loc}: unsupported workload_type '{record['workload_type']}'")

    for field in ("target_input_token_bucket", "expected_output_cap"):
        value = record[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"{loc}: field '{field}' must be a positive integer")

    shared_prefix_id = record["shared_prefix_id"]
    if shared_prefix_id is not None and not isinstance(shared_prefix_id, str):
        errors.append(f"{loc}: field 'shared_prefix_id' must be null or a string")

    if record["suite_type"] == "natural" and record["token_control_method"] != "not_applicable":
        errors.append(
            f"{loc}: natural prompts must use token_control_method='not_applicable'"
        )

    return errors


def load_prompt_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load prompt records from JSONL."""

    records: list[dict[str, Any]] = []
    prompt_path = Path(path)
    with prompt_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise PromptSchemaError(
                    f"{prompt_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise PromptSchemaError(
                    f"{prompt_path}:{line_number}: each JSONL row must be an object"
                )
            records.append(record)
    return records


def validate_prompt_records(
    records: list[dict[str, Any]], require_parallel_triplets: bool = True
) -> list[str]:
    """Return validation errors for a prompt dataset."""

    errors: list[str] = []
    seen_prompt_ids: set[str] = set()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for index, record in enumerate(records):
        errors.extend(validate_prompt_record(record, index))

        prompt_id = record.get("prompt_id")
        if isinstance(prompt_id, str):
            if prompt_id in seen_prompt_ids:
                errors.append(f"{prompt_id}: duplicate prompt_id")
            seen_prompt_ids.add(prompt_id)

        parallel_group_id = record.get("parallel_group_id")
        if isinstance(parallel_group_id, str):
            groups[parallel_group_id].append(record)

    if require_parallel_triplets:
        expected_languages = set(LANGUAGE_TO_SCRIPT)
        for group_id, group_records in sorted(groups.items()):
            languages = {record.get("language") for record in group_records}
            if len(group_records) != 3 or languages != expected_languages:
                errors.append(
                    f"{group_id}: expected exactly one hi, ta, and hinglish record"
                )

            for field in (
                "suite_type",
                "workload_type",
                "intent_category",
                "target_input_token_bucket",
                "expected_output_cap",
            ):
                values = {record.get(field) for record in group_records}
                if len(values) != 1:
                    errors.append(f"{group_id}: inconsistent field '{field}'")

    return errors


def assert_valid_prompt_records(
    records: list[dict[str, Any]], require_parallel_triplets: bool = True
) -> list[dict[str, Any]]:
    """Raise PromptSchemaError if records are invalid, otherwise return records."""

    errors = validate_prompt_records(records, require_parallel_triplets)
    if errors:
        raise PromptSchemaError("\n".join(errors))
    return records


def prompt_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Convert a prompt record into chat messages for tokenizer templates."""

    messages: list[dict[str, str]] = []
    system_prompt = record.get("system_prompt", "")
    context_text = record.get("context_text", "")
    user_prompt = record.get("user_prompt", "")

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = user_prompt
    if context_text:
        user_content = f"{context_text}\n\n{user_prompt}"

    messages.append({"role": "user", "content": user_content})
    return messages


def prompt_character_count(record: dict[str, Any]) -> int:
    """Count raw characters across system, context, and user prompt fields."""

    parts = [
        record.get("system_prompt", ""),
        record.get("context_text", ""),
        record.get("user_prompt", ""),
    ]
    return len("\n\n".join(part for part in parts if part))


def summarize_prompt_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a small summary useful for CLI validation output."""

    groups = {record["parallel_group_id"] for record in records}
    languages = {record["language"] for record in records}
    suite_types = {record["suite_type"] for record in records}
    workload_types = {record["workload_type"] for record in records}
    return {
        "record_count": len(records),
        "parallel_group_count": len(groups),
        "languages": sorted(languages),
        "suite_types": sorted(suite_types),
        "workload_types": sorted(workload_types),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate IndicServeBench prompts.")
    parser.add_argument("prompt_jsonl", type=Path)
    parser.add_argument(
        "--no-triplet-check",
        action="store_true",
        help="Disable the requirement that each parallel group contains hi/ta/hinglish.",
    )
    args = parser.parse_args()

    try:
        records = load_prompt_jsonl(args.prompt_jsonl)
        errors = validate_prompt_records(
            records, require_parallel_triplets=not args.no_triplet_check
        )
    except PromptSchemaError as exc:
        print(json.dumps({"status": "error", "errors": [str(exc)]}, indent=2))
        return 2

    if errors:
        print(json.dumps({"status": "error", "errors": errors}, indent=2))
        return 1

    print(
        json.dumps(
            {"status": "ok", "summary": summarize_prompt_records(records)},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
