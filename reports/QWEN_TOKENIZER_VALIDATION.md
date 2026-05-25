# Qwen Tokenizer Validation

Status: CPU-only tokenizer validation. No model weights, GPU serving, SGLang,
vLLM, or H100 commands were used.

## Purpose

This validation checks whether Qwen's high Hindi and Tamil token counts in the
prototype diagnostic are genuine tokenizer behavior rather than an FP8
checkpoint/config/accounting artifact.

English is included as a reference-control language. Hindi, Tamil, and Hinglish
remain the primary project workloads.

## Models Compared

- `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`
- `Qwen/Qwen3-30B-A3B-Instruct-2507`

## Raw Token ID Result

Across all 48 natural prompt records, the FP8 and non-FP8 Qwen checkpoints
produced identical raw user-prompt token IDs and counts.

- Differing prompt IDs: none
- Validation output: `results/tokenizer/qwen_tokenizer_validation_v1/validation.json`

## Raw-Token Accounting

`raw_user_prompt_tokens` is computed only from the raw `user_prompt` field with
`add_special_tokens=False`.

It does not include:

- `system_prompt`
- `context_text`
- role markers
- chat-template tokens

The summary label `formatted_minus_raw_user_tokens` is used in reports instead
of treating the difference as pure chat-template overhead, because formatted
input contains the system prompt, context text, user text, role wrappers,
generation prompt markers, and any model-specific chat-template tokens.

## Selected Fragment Counts

Selected prompt IDs:

- `nat_001_hi`
- `nat_001_ta`
- `nat_001_hinglish`
- `nat_001_en`

Raw user-prompt token counts:

| Prompt | Language | Sarvam | Qwen FP8 | Qwen non-FP8 |
| --- | ---: | ---: | ---: | ---: |
| `nat_001_hi` | hi | 20 | 75 | 75 |
| `nat_001_ta` | ta | 24 | 112 | 112 |
| `nat_001_hinglish` | hinglish | 21 | 30 | 30 |
| `nat_001_en` | en | 18 | 19 | 19 |

The selected examples show Qwen fragmenting native-script Hindi and Tamil much
more heavily than Sarvam, while English is closely matched. This is tokenizer
evidence only. It does not establish lower inference latency or higher model
quality.

GPU experiments are still required to test whether tokenizer differences affect
streaming TTFT or end-to-end serving behavior.
