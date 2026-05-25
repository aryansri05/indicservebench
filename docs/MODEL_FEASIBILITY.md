# Model Feasibility

Current status: planning and CPU-only preparation. No serving compatibility results exist yet.

## Candidate Matrix

| Model | Role | Architecture Notes | SGLang Status | vLLM Status | Context Status | Precision Status | Core Inclusion Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sarvamai/sarvam-30b` | Headline Indian-language model | TODO verify official architecture details, including MoE and active-parameter claims | Sarvam-specific smoke test planned; not yet run | Early common-runtime candidate; registry coverage noted; not yet run | TODO confirm exact context limit from config/docs | TODO verify BF16/FP8/quantization feasibility on one H100 | Pending |
| `krutrim-ai-labs/Krutrim-2-instruct` | Dense Indian-language instruction comparison | TODO verify dense 12B and model architecture details from official docs/config | Unknown until direct smoke test | Unknown until direct smoke test | TODO verify documented 128K support and serving behavior | TODO verify dtype, memory footprint, chat template, `trust_remote_code` | Pending |
| `bharatgenai/Param2-17B-A2.4B-Thinking` | Public Indian multilingual MoE comparison | TODO verify Hybrid MoE, 17B total, 2.4B active claims from official docs/config | Documented launch path requires smoke test | Early common-runtime candidate; vLLM supported-model coverage noted; not yet run | TODO verify current 4,096 context limit and safety margin | TODO verify dtype, memory footprint, thinking-mode behavior | Pending |

## Current Source Notes

- The NVIDIA/Sarvam public engineering article motivates a Sarvam-specific SGLang smoke test.
- Current vLLM documentation and registry coverage make vLLM an early common-runtime candidate.
- None of these source notes replace direct launch and streaming validation.

Sources:

- <https://developer.nvidia.com/blog/how-nvidia-extreme-hardware-software-co-design-delivered-a-large-inference-boost-for-sarvam-ais-sovereign-models/>
- <https://docs.vllm.ai/en/stable/models/supported_models/>
- <https://github.com/vllm-project/vllm/blob/main/tests/models/registry.py>
- <https://huggingface.co/sarvamai/sarvam-30b>
- <https://huggingface.co/krutrim-ai-labs/Krutrim-2-instruct>
- <https://huggingface.co/bharatgenai/Param2-17B-A2.4B-Thinking>

## Required Verification Tasks

Before any direct comparison:

- Confirm exact model revision/commit for each checkpoint.
- Confirm license and public deployability constraints.
- Confirm tokenizer and chat template behavior.
- Confirm whether `trust_remote_code` is needed.
- Confirm exact context limit from model config and runtime behavior.
- Confirm SGLang launch behavior where tested.
- Confirm vLLM launch behavior where tested.
- Confirm streaming output validity.
- Confirm deterministic or near-deterministic generation settings.
- Confirm precision/quantization feasibility on one H100.
- Confirm memory footprint for short context before long-context or concurrency tests.

## Inclusion Rule

A model can appear in the direct core benchmark only if it runs under the chosen common runtime with the same measurement procedure and compatible precision/quantization policy. A model that only works under a different runtime may be documented in an appendix, but must not be mixed into the direct comparison table.

## Long-Context Rule

The benchmark must validate:

```text
actual_formatted_input_tokens + maximum_requested_output_tokens <= confirmed_context_limit
```

This check is required for every model. It is especially important for BharatGen Param2 if its confirmed context limit remains 4,096 tokens.
