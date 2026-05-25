# IndicServeBench

IndicServeBench is a planned public benchmark for self-hosted streaming text-generation inference on Indian-language LLM workloads. The project focuses on Hindi in Devanagari, Tamil in Tamil script, and Hinglish in Latin script. English appears only as a reference-control language for tokenizer diagnostics and smoke-test sanity checks.

Current status: pre-GPU milestone. No benchmark results exist yet.

## Research Question

Under identical streaming-serving conditions on controlled GPU hardware, how do publicly deployable Indian-language LLMs differ in tokenizer cost, time to first token, inter-token latency, end-to-end latency, generated-token throughput, GPU memory pressure, and stability under concurrency?

## Scope

Version 1 is limited to self-hosted text-generation serving. It includes tokenizer analysis, prompt validation, runtime compatibility smoke testing, online streaming latency measurement, limited concurrency measurement, and result publication after the pre-GPU gates pass.

Version 1 does not include audio, speech-to-text, text-to-speech, proprietary APIs, fine-tuning, training, retrieval-augmented generation, model-quality ranking, dashboard development, or custom CUDA kernel optimization.

## Candidate Models

The v1 candidate model list is intentionally fixed:

- `sarvamai/sarvam-30b`
- `krutrim-ai-labs/Krutrim-2-instruct`
- `bharatgenai/Param2-17B-A2.4B-Thinking`

These models are not architecturally identical. Sarvam-30B and BharatGen Param2 are documented as MoE-family models, while Krutrim-2-Instruct is documented as a dense instruction model. Any future comparison must be framed as observed serving performance under the same test conditions, not as proof that one architecture is generally superior.

## Public Workload Inspiration

The long-context workload is inspired by a public NVIDIA/Sarvam engineering write-up that discusses Sarvam-30B inference optimization for real-time voice-agent-style workloads using SGLang, H100-class hardware, P95 TTFT, P95 ITL, an average input length of 3,584 tokens, and 128 output tokens.

Any future H100 results in this repository will be described as inspired by the reported workload shape. They will not be described as a reproduction of Sarvam/NVIDIA production performance.

Source: <https://developer.nvidia.com/blog/how-nvidia-extreme-hardware-software-co-design-delivered-a-large-inference-boost-for-sarvam-ais-sovereign-models/>

## Benchmark Tracks

1. Documentation and feasibility specification.
2. Tokenizer analysis.
3. Runtime compatibility smoke testing.
4. Single-request streaming latency.
5. Concurrent streaming serving.
6. Published cross-model report.
7. Optional T4-versus-H100 experiment only when the exact same configuration is feasible on both GPUs.

## Runtime Strategy

SGLang remains the first Sarvam-specific smoke-test runtime because the public NVIDIA/Sarvam study used SGLang. For direct cross-model comparison, the priority is a single common runtime across all included models.

vLLM is an early common-runtime candidate because current vLLM documentation/registry coverage includes BharatGen Param2 and vLLM registry entries include Sarvam-30B and BharatGen Param2. Krutrim runtime compatibility remains unknown until direct smoke testing. Final direct-comparison tables must use one common runtime only. A separate Sarvam-only SGLang appendix may be added later.

Sources:

- <https://docs.vllm.ai/en/stable/models/supported_models/>
- <https://github.com/vllm-project/vllm/blob/main/tests/models/registry.py>

## Pre-GPU Milestone

This milestone creates documentation, prompt schemas, prototype natural prompts, model configuration metadata, tokenizer-analysis tooling, and prompt-schema tests. It does not launch SGLang, vLLM, or any H100 instance.

CPU-only checks:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install transformers sentencepiece pyyaml pandas pytest jinja2
python3 src/indicservebench/prompt_schema.py prompts/prototype_prompts.jsonl
python -m pytest tests
python src/indicservebench/tokenizer_analysis.py
python src/indicservebench/token_fragment_inspect.py
```

Actual tokenizer counts require a Python environment with `transformers` and `jinja2` available. If `transformers`, a tokenizer, or a chat template cannot load, the tokenizer-analysis script records structured failure rows instead of crashing the whole run.

The tokenizer diagnostic writes each run to `results/tokenizer/<experiment_id>/` with `raw.jsonl`, `summary.csv`, and `metadata.json`. It records raw user-prompt token counts, formatted chat-template token counts, and template overhead. This diagnostic identifies tokenizer and chat-template differences. Actual serving latency will be measured later on GPU.

## Limitations

- No benchmark results exist yet.
- Runtime compatibility is not assumed until smoke-tested.
- Context-window values, precision choices, and memory footprints remain TODO verification items.
- Tokenizer analysis may require downloading tokenizer files, but must not download model weights.
