"""Controlled Sarvam-2B/T4 reproduction runner.

This runner is intended for Colab/Kaggle-style T4 execution only. The module is
importable on CPU machines for tests, but model loading happens only inside the
CLI run path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

try:
    from indicservebench.t4_reproduction_analysis import write_analysis_outputs
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from indicservebench.t4_reproduction_analysis import write_analysis_outputs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "prompts" / "sarvam2b_t4_original_prompts.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "t4_reproduction"

MODEL_ID = "sarvamai/sarvam-2b-v0.5"
BACKEND = "Hugging Face Transformers"
LANGUAGE_ORDER = ("hindi", "tamil", "codemix")
REPRODUCTION_TYPE = "approximate_reproduction"

PROMPT_WRAPPER = "<s>[INST] {prompt} [/INST]"
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_WARMUP_MAX_NEW_TOKENS = 10
DEFAULT_TEMPERATURE = 0.1
DEFAULT_DO_SAMPLE = True
DEFAULT_REPETITIONS = 5
DEFAULT_SEED = 42

RECOVERED_ORIGINAL_SETTINGS = {
    "model_id": MODEL_ID,
    "backend": BACKEND,
    "prompt_wrapper": PROMPT_WRAPPER,
    "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
    "warmup_inside_old_benchmark_max_new_tokens": DEFAULT_WARMUP_MAX_NEW_TOKENS,
    "temperature": DEFAULT_TEMPERATURE,
    "do_sample": DEFAULT_DO_SAMPLE,
    "pad_token_id": "tokenizer.eos_token_id",
    "tokenizer_trust_remote_code": True,
    "model_trust_remote_code": True,
    "dtype": "torch.float16",
    "device_map": "cuda",
}

UNRECOVERED_ORIGINAL_SETTINGS = (
    "random_seed",
    "per_request_raw_latency_rows",
    "per_request_generated_output_token_counts",
    "top_p_explicit_value",
    "top_k_explicit_value",
)

RAW_REQUEST_FIELDS = (
    "experiment_id",
    "reproduction_type",
    "timestamp_utc",
    "model_id",
    "backend",
    "gpu_name",
    "gpu_environment",
    "cuda_version",
    "torch_version",
    "transformers_version",
    "dtype",
    "generation_config",
    "random_seed",
    "prompt_id",
    "language",
    "repetition_id",
    "warmup_or_measured",
    "input_text_hash",
    "input_tokens",
    "output_tokens",
    "generated_text_hash",
    "total_latency_ms",
    "output_tokens_per_second",
    "peak_gpu_memory_mb",
    "success",
    "error_type",
    "error_message",
)


class PromptDatasetError(ValueError):
    """Raised when the recovered old prompt dataset is not valid."""


class T4EnvironmentError(RuntimeError):
    """Raised when a run would be mislabeled as a T4 reproduction."""


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def default_experiment_id(smoke_test: bool, repetitions: int, seed: int) -> str:
    mode = "smoke" if smoke_test else f"r{repetitions}"
    stamp = utc_now_iso().replace("-", "").replace(":", "").replace("+", "Z")
    return f"sarvam2b_t4_{mode}_seed{seed}_{stamp}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_recovered_prompts(path: Path = DEFAULT_PROMPTS_PATH) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise PromptDatasetError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise PromptDatasetError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)

    validate_recovered_prompts(records)
    return records


def validate_recovered_prompts(records: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    required = {
        "prompt_id",
        "language",
        "prompt_index",
        "text",
        "source_artifact",
        "prompt_wrapper",
    }
    seen_prompt_ids: set[str] = set()
    by_language: dict[str, list[dict[str, Any]]] = {language: [] for language in LANGUAGE_ORDER}

    for index, record in enumerate(records):
        missing = required - set(record)
        if missing:
            errors.append(f"record[{index}]: missing {sorted(missing)}")
            continue

        prompt_id = record["prompt_id"]
        language = record["language"]
        text = record["text"]
        prompt_wrapper = record["prompt_wrapper"]

        if not isinstance(prompt_id, str) or not prompt_id:
            errors.append(f"record[{index}]: prompt_id must be a non-empty string")
        elif prompt_id in seen_prompt_ids:
            errors.append(f"{prompt_id}: duplicate prompt_id")
        seen_prompt_ids.add(str(prompt_id))

        if language not in LANGUAGE_ORDER:
            errors.append(f"{prompt_id}: unsupported language {language!r}")
        else:
            by_language[language].append(record)

        if not isinstance(text, str) or not text.strip():
            errors.append(f"{prompt_id}: text must be a non-empty string")

        if prompt_wrapper != PROMPT_WRAPPER:
            errors.append(f"{prompt_id}: prompt_wrapper does not match recovered wrapper")

        prompt_index = record["prompt_index"]
        if not isinstance(prompt_index, int) or isinstance(prompt_index, bool):
            errors.append(f"{prompt_id}: prompt_index must be an integer")

    if len(records) != 30:
        errors.append(f"expected 30 recovered prompts, found {len(records)}")

    for language in LANGUAGE_ORDER:
        language_records = by_language[language]
        indexes = sorted(record.get("prompt_index") for record in language_records)
        if len(language_records) != 10:
            errors.append(f"{language}: expected 10 prompts, found {len(language_records)}")
        if indexes != list(range(1, 11)):
            errors.append(f"{language}: expected prompt indexes 1..10, found {indexes}")

    if errors:
        raise PromptDatasetError("\n".join(errors))


def format_prompt(record: dict[str, Any]) -> str:
    return PROMPT_WRAPPER.format(prompt=record["text"])


def select_first_prompt_per_language(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for language in LANGUAGE_ORDER:
        candidates = sorted(
            (record for record in records if record["language"] == language),
            key=lambda record: record["prompt_index"],
        )
        if not candidates:
            raise PromptDatasetError(f"missing prompt for language {language}")
        selected.append(candidates[0])
    return selected


def build_warmup_requests(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for record in select_first_prompt_per_language(records):
        requests.append(
            {
                "prompt": record,
                "repetition_id": 0,
                "warmup_or_measured": "warmup",
            }
        )
    return requests


def build_measured_requests(
    records: list[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
    smoke_test: bool,
) -> list[dict[str, Any]]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    if smoke_test:
        base_records = select_first_prompt_per_language(records)
        requests = [
            {
                "prompt": record,
                "repetition_id": 1,
                "warmup_or_measured": "measured",
            }
            for record in base_records
        ]
    else:
        requests = []
        for repetition_id in range(1, repetitions + 1):
            for record in records:
                requests.append(
                    {
                        "prompt": record,
                        "repetition_id": repetition_id,
                        "warmup_or_measured": "measured",
                    }
                )

    random.Random(seed).shuffle(requests)
    return requests


def validate_t4_environment(gpu_name: str | None, allow_non_t4: bool) -> str:
    if gpu_name and "T4" in gpu_name.upper():
        return "t4_environment"
    if allow_non_t4:
        return "non_t4_environment"
    if not gpu_name:
        raise T4EnvironmentError(
            "No CUDA GPU was detected. Use --allow-non-t4 only for explicitly labeled "
            "non-T4 dry/diagnostic environments."
        )
    raise T4EnvironmentError(
        f"Detected GPU {gpu_name!r}, not a T4. Use --allow-non-t4 only if you want "
        "the output labeled as non_t4_environment."
    )


def token_count_from_inputs(inputs: Any) -> int:
    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    shape = getattr(input_ids, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[1])
    if isinstance(input_ids, list):
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)
    raise ValueError(f"unsupported input_ids shape/type: {type(input_ids).__name__}")


def generation_config_for_row(
    *,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: float | None,
    top_k: int | None,
    pad_token_id: str,
) -> dict[str, Any]:
    return {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "top_p": top_p,
        "top_k": top_k,
        "pad_token_id": pad_token_id,
    }


def generation_kwargs(
    tokenizer: Any,
    *,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: float | None,
    top_k: int | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None:
        kwargs["top_k"] = top_k
    return kwargs


def make_raw_record(
    *,
    experiment_id: str,
    gpu_name: str | None,
    gpu_environment: str,
    cuda_version: str | None,
    torch_version: str | None,
    transformers_version: str | None,
    dtype: str,
    generation_config: dict[str, Any],
    random_seed: int,
    prompt_record: dict[str, Any],
    repetition_id: int,
    warmup_or_measured: str,
    input_text_hash: str,
    input_tokens: int | None,
    output_tokens: int | None,
    generated_text_hash: str | None,
    total_latency_ms: float | None,
    output_tokens_per_second: float | None,
    peak_gpu_memory_mb: float | None,
    success: bool,
    error_type: str | None,
    error_message: str | None,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "reproduction_type": REPRODUCTION_TYPE,
        "timestamp_utc": utc_now_iso(),
        "model_id": MODEL_ID,
        "backend": BACKEND,
        "gpu_name": gpu_name,
        "gpu_environment": gpu_environment,
        "cuda_version": cuda_version,
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "dtype": dtype,
        "generation_config": generation_config,
        "random_seed": random_seed,
        "prompt_id": prompt_record["prompt_id"],
        "language": prompt_record["language"],
        "repetition_id": repetition_id,
        "warmup_or_measured": warmup_or_measured,
        "input_text_hash": input_text_hash,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "generated_text_hash": generated_text_hash,
        "total_latency_ms": total_latency_ms,
        "output_tokens_per_second": output_tokens_per_second,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "success": success,
        "error_type": error_type,
        "error_message": error_message,
    }


def validate_raw_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in RAW_REQUEST_FIELDS:
        if field not in record:
            errors.append(f"missing field '{field}'")
    if errors:
        return errors

    if record["reproduction_type"] not in {"exact_reproduction", "approximate_reproduction"}:
        errors.append("reproduction_type must be exact_reproduction or approximate_reproduction")
    if record["model_id"] != MODEL_ID:
        errors.append(f"model_id must be {MODEL_ID}")
    if record["backend"] != BACKEND:
        errors.append(f"backend must be {BACKEND}")
    if record["language"] not in LANGUAGE_ORDER:
        errors.append(f"unsupported language {record['language']!r}")
    if record["warmup_or_measured"] not in {"warmup", "measured"}:
        errors.append("warmup_or_measured must be warmup or measured")
    if not isinstance(record["generation_config"], dict):
        errors.append("generation_config must be an object")
    if not isinstance(record["success"], bool):
        errors.append("success must be boolean")

    nullable_numbers = (
        "input_tokens",
        "output_tokens",
        "total_latency_ms",
        "output_tokens_per_second",
        "peak_gpu_memory_mb",
    )
    for field in nullable_numbers:
        value = record[field]
        if value is not None and not isinstance(value, (int, float)):
            errors.append(f"{field} must be numeric or null")

    if record["error_type"] is not None and not isinstance(record["error_type"], str):
        errors.append("error_type must be a string or null")
    if record["error_message"] is not None and not isinstance(record["error_message"], str):
        errors.append("error_message must be a string or null")

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


def seed_runtime(torch_module: Any, seed: int) -> None:
    random.seed(seed)
    if hasattr(torch_module, "manual_seed"):
        torch_module.manual_seed(seed)
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and hasattr(cuda, "manual_seed_all"):
        cuda.manual_seed_all(seed)


def dtype_from_name(torch_module: Any, dtype_name: str) -> Any:
    mapping = {
        "float16": getattr(torch_module, "float16"),
        "bfloat16": getattr(torch_module, "bfloat16", None),
        "float32": getattr(torch_module, "float32"),
    }
    dtype = mapping.get(dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported dtype {dtype_name!r}")
    return dtype


def load_runtime_modules() -> tuple[Any, Any, Any, Any]:
    import torch  # type: ignore
    import transformers  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    return torch, transformers, AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    *,
    AutoModelForCausalLM: Any,
    AutoTokenizer: Any,
    torch_module: Any,
    dtype_name: str,
    device_map: str,
    local_files_only: bool,
) -> tuple[Any, Any]:
    dtype_value = dtype_from_name(torch_module, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
        "local_files_only": local_files_only,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=dtype_value,
            **model_kwargs,
        )
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype_value,
            **model_kwargs,
        )
    model.eval()
    return model, tokenizer


def run_one_request(
    *,
    torch_module: Any,
    model: Any,
    tokenizer: Any,
    request: dict[str, Any],
    experiment_id: str,
    gpu_name: str | None,
    gpu_environment: str,
    cuda_version: str | None,
    torch_version: str | None,
    transformers_version: str | None,
    dtype: str,
    random_seed: int,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: float | None,
    top_k: int | None,
) -> dict[str, Any]:
    prompt_record = request["prompt"]
    formatted_input = format_prompt(prompt_record)
    input_hash = sha256_text(formatted_input)
    config_for_row = generation_config_for_row(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        pad_token_id="tokenizer.eos_token_id",
    )

    input_tokens: int | None = None
    output_tokens: int | None = None
    generated_text_hash: str | None = None
    total_latency_ms: float | None = None
    output_tokens_per_second: float | None = None
    peak_gpu_memory_mb: float | None = None
    success = False
    error_type: str | None = None
    error_message: str | None = None

    try:
        inputs = tokenizer(formatted_input, return_tensors="pt")
        input_tokens = token_count_from_inputs(inputs)
        if hasattr(inputs, "to"):
            inputs = inputs.to("cuda")
        else:
            inputs = {
                key: value.to("cuda") if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

        cuda = torch_module.cuda
        if hasattr(cuda, "reset_peak_memory_stats"):
            cuda.reset_peak_memory_stats()
        cuda.synchronize()
        start_ns = time.perf_counter_ns()
        with torch_module.no_grad():
            output = model.generate(
                **inputs,
                **generation_kwargs(
                    tokenizer,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                ),
            )
        cuda.synchronize()
        end_ns = time.perf_counter_ns()

        total_latency_ms = (end_ns - start_ns) / 1_000_000
        output_tokens = int(output.shape[1] - input_tokens)
        generated_ids = output[0][input_tokens:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_text_hash = sha256_text(generated_text)
        if total_latency_ms > 0:
            output_tokens_per_second = output_tokens / (total_latency_ms / 1000)
        if hasattr(cuda, "max_memory_allocated"):
            peak_gpu_memory_mb = cuda.max_memory_allocated() / (1024 * 1024)
        success = True
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)

    return make_raw_record(
        experiment_id=experiment_id,
        gpu_name=gpu_name,
        gpu_environment=gpu_environment,
        cuda_version=cuda_version,
        torch_version=torch_version,
        transformers_version=transformers_version,
        dtype=dtype,
        generation_config=config_for_row,
        random_seed=random_seed,
        prompt_record=prompt_record,
        repetition_id=request["repetition_id"],
        warmup_or_measured=request["warmup_or_measured"],
        input_text_hash=input_hash,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        generated_text_hash=generated_text_hash,
        total_latency_ms=total_latency_ms,
        output_tokens_per_second=output_tokens_per_second,
        peak_gpu_memory_mb=peak_gpu_memory_mb,
        success=success,
        error_type=error_type,
        error_message=error_message,
    )


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_metadata(
    *,
    args: argparse.Namespace,
    experiment_id: str,
    output_dir: Path,
    raw_jsonl: Path,
    prompt_count: int,
    measured_request_count: int,
    warmup_request_count: int,
    gpu_name: str | None,
    gpu_environment: str,
    cuda_version: str | None,
    torch_version: str | None,
    transformers_version: str | None,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "created_at_utc": utc_now_iso(),
        "reproduction_type": REPRODUCTION_TYPE,
        "model_id": MODEL_ID,
        "backend": BACKEND,
        "gpu_name": gpu_name,
        "gpu_environment": gpu_environment,
        "cuda_version": cuda_version,
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "prompt_dataset_path": str(args.prompts),
        "prompt_dataset_sha256": sha256_file(args.prompts),
        "prompt_count": prompt_count,
        "warmup_request_count": warmup_request_count,
        "measured_request_count": measured_request_count,
        "raw_requests_jsonl": str(raw_jsonl),
        "output_dir": str(output_dir),
        "runner_cli_args": sys.argv[1:],
        "random_seed": args.seed,
        "repetitions": 1 if args.smoke_test else args.repetitions,
        "smoke_test": args.smoke_test,
        "recovered_original_settings": RECOVERED_ORIGINAL_SETTINGS,
        "unrecovered_original_settings": list(UNRECOVERED_ORIGINAL_SETTINGS),
        "controlled_generation_settings": generation_config_for_row(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            pad_token_id="tokenizer.eos_token_id",
        ),
        "warmup_generation_settings": generation_config_for_row(
            max_new_tokens=args.warmup_max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            pad_token_id="tokenizer.eos_token_id",
        ),
        "raw_request_schema": list(RAW_REQUEST_FIELDS),
        "interpretation_guardrail": (
            "This runner logs request-level latency and token counts. Results must "
            "not be interpreted as CUDA/kernel bottleneck evidence by themselves."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the controlled Sarvam-2B/T4 reproduction with raw logging."
    )
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-non-t4", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--warmup-max-new-tokens",
        type=int,
        default=DEFAULT_WARMUP_MAX_NEW_TOKENS,
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=True)
    parser.add_argument("--no-do-sample", dest="do_sample", action="store_false")
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--device-map", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_recovered_prompts(args.prompts)

    torch_module, transformers_module, AutoModelForCausalLM, AutoTokenizer = (
        load_runtime_modules()
    )
    if not torch_module.cuda.is_available():
        gpu_name = None
    else:
        gpu_name = torch_module.cuda.get_device_name(0)
    gpu_environment = validate_t4_environment(gpu_name, args.allow_non_t4)

    seed_runtime(torch_module, args.seed)

    warmup_requests = build_warmup_requests(records)
    measured_requests = build_measured_requests(
        records,
        repetitions=args.repetitions,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )
    experiment_id = args.experiment_id or default_experiment_id(
        args.smoke_test, args.repetitions, args.seed
    )
    output_dir = args.output_dir / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = output_dir / "raw_requests.jsonl"
    metadata_json = output_dir / "metadata.json"
    if raw_jsonl.exists():
        raise FileExistsError(
            f"{raw_jsonl} already exists. Use a new --experiment-id to avoid mixing runs."
        )

    model, tokenizer = load_model_and_tokenizer(
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
        torch_module=torch_module,
        dtype_name=args.dtype,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
    )

    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    torch_version = getattr(torch_module, "__version__", None)
    transformers_version = getattr(transformers_module, "__version__", None)

    metadata = build_metadata(
        args=args,
        experiment_id=experiment_id,
        output_dir=output_dir,
        raw_jsonl=raw_jsonl,
        prompt_count=len(records),
        measured_request_count=len(measured_requests),
        warmup_request_count=len(warmup_requests),
        gpu_name=gpu_name,
        gpu_environment=gpu_environment,
        cuda_version=cuda_version,
        torch_version=torch_version,
        transformers_version=transformers_version,
    )
    write_metadata(metadata_json, metadata)

    for request in warmup_requests + measured_requests:
        max_new_tokens = (
            args.warmup_max_new_tokens
            if request["warmup_or_measured"] == "warmup"
            else args.max_new_tokens
        )
        row = run_one_request(
            torch_module=torch_module,
            model=model,
            tokenizer=tokenizer,
            request=request,
            experiment_id=experiment_id,
            gpu_name=gpu_name,
            gpu_environment=gpu_environment,
            cuda_version=cuda_version,
            torch_version=torch_version,
            transformers_version=transformers_version,
            dtype=args.dtype,
            random_seed=args.seed,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        append_jsonl(raw_jsonl, row)

    analysis_result = write_analysis_outputs(raw_jsonl, output_dir)
    print(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "reproduction_type": REPRODUCTION_TYPE,
                "gpu_name": gpu_name,
                "gpu_environment": gpu_environment,
                "output_dir": str(output_dir),
                "raw_requests_jsonl": str(raw_jsonl),
                **analysis_result,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
