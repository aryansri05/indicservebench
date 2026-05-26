"""Analysis helpers for the Sarvam-2B/T4 controlled reproduction.

This module is CPU-only. It reads raw JSONL rows produced by the reproduction
runner and writes aggregate CSV artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


LANGUAGE_ORDER = ("hindi", "tamil", "codemix")
QUANTILE_METHOD = "linear_interpolation_position_(n_minus_1)*q"

AGGREGATE_FIELDS = (
    "language",
    "measured_request_count",
    "successful_request_count",
    "failed_request_count",
    "mean_input_tokens",
    "median_input_tokens",
    "mean_output_tokens",
    "median_output_tokens",
    "median_latency_ms",
    "p90_latency_ms",
    "p95_latency_ms",
    "max_latency_ms",
    "mean_output_tokens_per_second",
    "failure_rate",
)

SLOWEST_FIELDS = (
    "language",
    "prompt_id",
    "repetition_id",
    "input_tokens",
    "output_tokens",
    "total_latency_ms",
    "output_tokens_per_second",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def quantile(values: list[float], q: float) -> float | None:
    """Return a linear-interpolated quantile using position (n - 1) * q."""

    if not values:
        return None
    if not 0 <= q <= 1:
        raise ValueError("q must be between 0 and 1")

    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * q
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    upper_weight = position - lower_index
    lower_weight = 1.0 - upper_weight
    return lower_value * lower_weight + upper_value * upper_weight


def _numeric_values(records: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _rounded(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate measured rows by language, excluding warmups."""

    measured = [
        record for record in records if record.get("warmup_or_measured") == "measured"
    ]
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in measured:
        by_language[str(record.get("language"))].append(record)

    rows: list[dict[str, Any]] = []
    for language in LANGUAGE_ORDER:
        language_rows = by_language.get(language, [])
        success_rows = [record for record in language_rows if record.get("success") is True]
        failed_request_count = len(language_rows) - len(success_rows)

        input_tokens = _numeric_values(success_rows, "input_tokens")
        output_tokens = _numeric_values(success_rows, "output_tokens")
        latencies = _numeric_values(success_rows, "total_latency_ms")
        tokens_per_second = _numeric_values(success_rows, "output_tokens_per_second")

        row = {
            "language": language,
            "measured_request_count": len(language_rows),
            "successful_request_count": len(success_rows),
            "failed_request_count": failed_request_count,
            "mean_input_tokens": _rounded(mean(input_tokens)) if input_tokens else None,
            "median_input_tokens": _rounded(median(input_tokens)) if input_tokens else None,
            "mean_output_tokens": _rounded(mean(output_tokens)) if output_tokens else None,
            "median_output_tokens": _rounded(median(output_tokens)) if output_tokens else None,
            "median_latency_ms": _rounded(median(latencies)) if latencies else None,
            "p90_latency_ms": _rounded(quantile(latencies, 0.90)),
            "p95_latency_ms": _rounded(quantile(latencies, 0.95)),
            "max_latency_ms": _rounded(max(latencies)) if latencies else None,
            "mean_output_tokens_per_second": (
                _rounded(mean(tokens_per_second)) if tokens_per_second else None
            ),
            "failure_rate": (
                _rounded(failed_request_count / len(language_rows))
                if language_rows
                else None
            ),
        }
        rows.append(row)

    return rows


def select_slowest_requests(
    records: list[dict[str, Any]], per_language: int = 5
) -> list[dict[str, Any]]:
    measured_successes = [
        record
        for record in records
        if record.get("warmup_or_measured") == "measured"
        and record.get("success") is True
        and isinstance(record.get("total_latency_ms"), (int, float))
    ]
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in measured_successes:
        by_language[str(record.get("language"))].append(record)

    slowest: list[dict[str, Any]] = []
    for language in LANGUAGE_ORDER:
        language_rows = sorted(
            by_language.get(language, []),
            key=lambda record: float(record["total_latency_ms"]),
            reverse=True,
        )[:per_language]
        for record in language_rows:
            slowest.append(
                {
                    "language": language,
                    "prompt_id": record.get("prompt_id"),
                    "repetition_id": record.get("repetition_id"),
                    "input_tokens": record.get("input_tokens"),
                    "output_tokens": record.get("output_tokens"),
                    "total_latency_ms": record.get("total_latency_ms"),
                    "output_tokens_per_second": record.get("output_tokens_per_second"),
                }
            )
    return slowest


def repeated_prompt_ids_in_slowest(
    slowest_rows: list[dict[str, Any]], language: str = "tamil"
) -> dict[str, int]:
    prompt_ids = [
        str(record["prompt_id"])
        for record in slowest_rows
        if record.get("language") == language and record.get("prompt_id")
    ]
    counts = Counter(prompt_ids)
    return {prompt_id: count for prompt_id, count in counts.items() if count > 1}


def _mean_for_language(
    records: list[dict[str, Any]],
    language: str,
    field: str,
    *,
    measured_success_only: bool = True,
) -> float | None:
    subset = [record for record in records if record.get("language") == language]
    if measured_success_only:
        subset = [
            record
            for record in subset
            if record.get("warmup_or_measured") == "measured"
            and record.get("success") is True
        ]
    values = _numeric_values(subset, field)
    if not values:
        return None
    return float(mean(values))


def build_analysis_notes(
    records: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    slowest_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    threshold_ratio = 1.10
    tail_reproduction_threshold_ratio = 1.25
    tamil_slowest = [row for row in slowest_rows if row.get("language") == "tamil"]

    tamil_slowest_input_mean = (
        mean(_numeric_values(tamil_slowest, "input_tokens")) if tamil_slowest else None
    )
    tamil_slowest_output_mean = (
        mean(_numeric_values(tamil_slowest, "output_tokens")) if tamil_slowest else None
    )
    hindi_input_mean = _mean_for_language(records, "hindi", "input_tokens")
    hindi_output_mean = _mean_for_language(records, "hindi", "output_tokens")

    aggregate_by_language = {row["language"]: row for row in aggregate_rows}
    tamil_p95 = aggregate_by_language.get("tamil", {}).get("p95_latency_ms")
    hindi_p95 = aggregate_by_language.get("hindi", {}).get("p95_latency_ms")
    codemix_p95 = aggregate_by_language.get("codemix", {}).get("p95_latency_ms")

    old_tail_reproduced = False
    if all(isinstance(value, (int, float)) for value in (tamil_p95, hindi_p95, codemix_p95)):
        old_tail_reproduced = bool(
            tamil_p95 > hindi_p95 * tail_reproduction_threshold_ratio
            and tamil_p95 > codemix_p95 * tail_reproduction_threshold_ratio
        )

    input_materially_larger = False
    if tamil_slowest_input_mean is not None and hindi_input_mean:
        input_materially_larger = tamil_slowest_input_mean > hindi_input_mean * threshold_ratio

    output_materially_larger = False
    if tamil_slowest_output_mean is not None and hindi_output_mean:
        output_materially_larger = tamil_slowest_output_mean > hindi_output_mean * threshold_ratio

    return {
        "quantile_method": QUANTILE_METHOD,
        "materiality_threshold_ratio": threshold_ratio,
        "tail_reproduction_threshold_ratio": tail_reproduction_threshold_ratio,
        "repeated_tamil_prompt_ids_among_slowest": repeated_prompt_ids_in_slowest(
            slowest_rows, "tamil"
        ),
        "tamil_slowest_mean_input_tokens": _rounded(tamil_slowest_input_mean),
        "hindi_measured_mean_input_tokens": _rounded(hindi_input_mean),
        "tamil_slowest_input_tokens_materially_larger_than_hindi_mean": input_materially_larger,
        "tamil_slowest_mean_output_tokens": _rounded(tamil_slowest_output_mean),
        "hindi_measured_mean_output_tokens": _rounded(hindi_output_mean),
        "tamil_slowest_output_tokens_materially_larger_than_hindi_mean": output_materially_larger,
        "old_tamil_tail_observation_appears_to_reproduce_by_threshold": old_tail_reproduced,
        "interpretation_guardrail": (
            "This analysis can identify repeated slow prompts and token-count "
            "relationships, but it does not establish a CUDA/kernel bottleneck."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def update_metadata(path: Path, analysis_notes: dict[str, Any]) -> None:
    metadata: dict[str, Any] = {}
    if path.exists():
        metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["analysis"] = analysis_notes
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_analysis_outputs(raw_jsonl: Path, output_dir: Path) -> dict[str, Any]:
    records = read_jsonl(raw_jsonl)
    aggregate_rows = aggregate_records(records)
    slowest_rows = select_slowest_requests(records)
    analysis_notes = build_analysis_notes(records, aggregate_rows, slowest_rows)

    write_csv(output_dir / "aggregate_summary.csv", aggregate_rows, AGGREGATE_FIELDS)
    write_csv(output_dir / "slowest_requests.csv", slowest_rows, SLOWEST_FIELDS)
    update_metadata(output_dir / "metadata.json", analysis_notes)

    return {
        "aggregate_summary_csv": str(output_dir / "aggregate_summary.csv"),
        "slowest_requests_csv": str(output_dir / "slowest_requests.csv"),
        "analysis": analysis_notes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Sarvam-2B/T4 reproduction raw JSONL results."
    )
    parser.add_argument("--raw-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = write_analysis_outputs(args.raw_jsonl, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
