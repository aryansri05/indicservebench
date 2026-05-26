# Run Sarvam-2B T4 Controlled Reproduction

This document describes how to rerun the earlier Sarvam-2B/T4 latency
experiment with request-level logging. It is intentionally narrow:

- model: `sarvamai/sarvam-2b-v0.5`
- backend: Hugging Face Transformers
- target GPU: one NVIDIA T4
- prompts: exact recovered original Hindi, Tamil, and code-mixed prompts

It does not use Sarvam-30B, Qwen, BharatGen, Krutrim, SGLang, vLLM, H100,
concurrency, long-context prompts, charts, dashboards, or kernel optimization
experiments.

## Purpose

The goal is to determine whether the earlier Tamil tail-latency observation
reproduces when the same basic setup is rerun with:

- raw per-request latency logging;
- input-token and output-token logging;
- repeated measurements;
- shuffled measured request order;
- warmups excluded from aggregate statistics.

The analysis should test whether slow Tamil requests are associated with input
token count, output token count, a repeated prompt ID, or general runtime
variance. It must not be interpreted as evidence of a CUDA/kernel bottleneck by
itself.

## Old Result

The preserved aggregate result was:

| Language | Median latency ms | Reported P95 ms | Mean tok/s |
|---|---:|---:|---:|
| Hindi | 2471.2 | 3099.1 | 24.8 |
| Tamil | 2535.9 | 5477.0 | 22.5 |
| Code-mixed Hindi-English | 2468.5 | 2996.7 | 25.0 |

This old result is inconclusive because:

- only 10 samples per language were measured;
- raw per-request rows were not preserved;
- generated output-token counts were not preserved;
- the reported P95 selected the maximum observation because `n=10`;
- CPU-only reconciliation found only a small Tamil-vs-Hindi input-token gap.

## Recovery Status

Exact original prompts were recovered from:

- `/Users/aryansrivastava/Downloads/sarvam benchmarks collab (1).ipynb`, cell 52

They are now preserved in:

- `prompts/sarvam2b_t4_original_prompts.jsonl`

Recovered original settings:

- prompt wrapper: `<s>[INST] {prompt} [/INST]`
- model loading: `AutoModelForCausalLM.from_pretrained(...)`
- tokenizer/model `trust_remote_code=True`
- dtype: `torch.float16`
- device map: `cuda`
- measured `max_new_tokens=64`
- old internal warmup `max_new_tokens=10`
- `temperature=0.1`
- `do_sample=True`
- `pad_token_id=tokenizer.eos_token_id`

Unrecovered or not preserved:

- random seed;
- raw per-request latency rows;
- raw per-request generated output-token counts;
- explicit `top_p`;
- explicit `top_k`.

Because the seed and raw rows were not preserved, and because this controlled
runner deliberately adds repetitions and shuffled request order, new runs are
labeled `approximate_reproduction`, not `exact_reproduction`.

## Colab T4 Setup

1. In Colab, choose `Runtime > Change runtime type > T4 GPU`.

2. Confirm the GPU:

```bash
!nvidia-smi
```

The runner refuses to label a non-T4 run as a T4 reproduction unless
`--allow-non-t4` is explicitly passed. Do not use that override for the main
reproduction.

3. Clone the repository:

```bash
!git clone https://github.com/aryansri05/indicservebench.git
%cd indicservebench
```

4. Install dependencies:

```bash
!python -m pip install -U transformers accelerate sentencepiece protobuf pandas pytest
```

5. Run tests:

```bash
!PYTHONPATH=src python -m pytest
```

6. Run a smoke test before any full reproduction:

```bash
!PYTHONPATH=src python -m indicservebench.t4_reproduction_runner \
  --smoke-test \
  --seed 42 \
  --output-dir results/t4_reproduction \
  --experiment-id sarvam2b_t4_smoke_seed42
```

Smoke-test scope:

- 3 warmup requests: one Hindi, one Tamil, one code-mixed prompt;
- 3 measured requests: one Hindi, one Tamil, one code-mixed prompt;
- measured output cap: 64 tokens;
- warmup output cap: 10 tokens;
- shuffled measured order;
- raw rows appended immediately to JSONL.

7. Inspect the smoke-test raw JSONL:

```bash
!head -n 3 results/t4_reproduction/sarvam2b_t4_smoke_seed42/raw_requests.jsonl
!cat results/t4_reproduction/sarvam2b_t4_smoke_seed42/aggregate_summary.csv
```

8. If the smoke test succeeds, run the 5-repetition reproduction:

```bash
!PYTHONPATH=src python -m indicservebench.t4_reproduction_runner \
  --repetitions 5 \
  --seed 42 \
  --output-dir results/t4_reproduction \
  --experiment-id sarvam2b_t4_r5_seed42
```

This produces 3 warmup rows and 150 measured rows:

- 10 prompts per language;
- 3 languages;
- 5 measured repetitions per prompt.

9. Inspect aggregate and slowest-request artifacts:

```bash
!cat results/t4_reproduction/sarvam2b_t4_r5_seed42/aggregate_summary.csv
!cat results/t4_reproduction/sarvam2b_t4_r5_seed42/slowest_requests.csv
!cat results/t4_reproduction/sarvam2b_t4_r5_seed42/metadata.json
```

10. Optionally run 10 repetitions only after reviewing the 5-repetition result:

```bash
!PYTHONPATH=src python -m indicservebench.t4_reproduction_runner \
  --repetitions 10 \
  --seed 42 \
  --output-dir results/t4_reproduction \
  --experiment-id sarvam2b_t4_r10_seed42
```

11. Download result artifacts:

```bash
!zip -r sarvam2b_t4_r5_seed42_results.zip results/t4_reproduction/sarvam2b_t4_r5_seed42
from google.colab import files
files.download("sarvam2b_t4_r5_seed42_results.zip")
```

## Output Files

Each run writes:

```text
results/t4_reproduction/<experiment_id>/
  raw_requests.jsonl
  aggregate_summary.csv
  slowest_requests.csv
  metadata.json
```

`raw_requests.jsonl` is append-only during the run. Every warmup, measured
request, and failure is recorded immediately.

Use a fresh `--experiment-id` for each run. The runner refuses to append to an
existing `raw_requests.jsonl` so separate attempts are not accidentally mixed.

## Quantile Method

The new aggregation uses linear interpolation with position `(n - 1) * q`.
This intentionally avoids repeating the old max-as-P95 behavior. The old
reported P95 is treated as the maximum observation for a 10-sample group.

## Interpretation Guidance

Reproducing a Tamil tail does not establish a CUDA/kernel bottleneck. It means
the controlled run should be inspected for:

- repeated Tamil prompt IDs among the slowest rows;
- larger input-token counts among slow Tamil rows;
- larger output-token counts among slow Tamil rows;
- stochastic/runtime variance.

Not reproducing the Tamil tail means the old observation should be downgraded
to a small, unreproducible aggregate observation unless additional raw evidence
is found.

Output-token counts must be checked before attributing latency differences to
language or serving behavior.
