# Benchmark Specification

Current status: pre-GPU specification. No benchmark results exist yet.

## Objective

IndicServeBench measures self-hosted streaming text-generation serving behavior for public Indian-language LLMs across Hindi, Tamil, and Hinglish workloads.

The project is focused on the LLM-serving layer of conversational and voice-agent-style systems. It is not an audio benchmark, ASR benchmark, TTS benchmark, proprietary API benchmark, fine-tuning project, model-quality leaderboard, RAG system, or chatbot product.

## Candidate Models

The v1 candidate set is fixed:

- `sarvamai/sarvam-30b`
- `krutrim-ai-labs/Krutrim-2-instruct`
- `bharatgenai/Param2-17B-A2.4B-Thinking`

No extra models are part of v1 unless the project scope is explicitly revised.

## Language Groups

The primary benchmark contains exactly three Indic workload groups:

- Hindi: Devanagari script.
- Tamil: Tamil script.
- Hinglish: Hindi-English code-mixed text primarily in Latin script.

English is included only as `reference_control` for tokenizer diagnostics and
smoke-test sanity checks. It is not a primary Indic workload and must not change
the headline v1 scope.

## Prompt Suites

### Suite A: Natural Semantic-Equivalent Prompts

Purpose: measure realistic user-facing serving behavior, including natural tokenizer differences.

Each `parallel_group_id` contains Hindi, Tamil, Hinglish, and English reference-control versions of the same intent. The wording should be natural in each language, not a literal translation. Suite A is not token-controlled. If one language produces more formatted tokens, higher latency may partly reflect tokenizer behavior.

The current repository contains only prototype Suite A prompts in `prompts/prototype_prompts.jsonl`.

Required prompt JSONL fields:

- `schema_version`
- `prompt_id`
- `parallel_group_id`
- `language`
- `script`
- `suite_type`
- `workload_type`
- `intent_category`
- `system_prompt`
- `context_text`
- `user_prompt`
- `target_input_token_bucket`
- `expected_output_cap`
- `shared_prefix_id`
- `token_control_method`
- `notes`

### Suite B: Token-Controlled Workloads

Purpose: compare runtime behavior when formatted input-token load is approximately equal.

This phase is intentionally not implemented yet. The design requirements are:

- Apply each model's own chat template before counting tokens.
- Count formatted input tokens after templating.
- Construct context from reusable natural blocks rather than malformed padding.
- Keep the final user task semantically comparable across languages.
- Reject any prompt that exceeds model context constraints.
- Never silently truncate.

## Workload Buckets

Input-token length is measured after model-specific chat-template formatting.

- Short chat: approximately 128 formatted input tokens, output cap 128.
- Contextual support chat: approximately 1,024 formatted input tokens, output cap 128.
- Voice-agent-style long context: target 3,584 formatted input tokens, output cap 128.

The long-context target is inspired by the public NVIDIA/Sarvam workload shape. It is not a reproduction of Sarvam/NVIDIA production performance.

## Long-Context Safety Rule

For every model, the harness must calculate formatted input tokens after applying the model-specific chat template. A request is valid only when:

```text
actual_formatted_input_tokens + maximum_requested_output_tokens <= confirmed_context_limit
```

If the confirmed context limit is unknown, long-context tests are blocked for that model. If the check fails, the prompt must be rejected and logged. It must not be truncated silently.

For BharatGen Param2, the currently documented maximum context is treated as a TODO verification item. If it remains 4,096 tokens, the 3,584-input/128-output case has limited margin and must be validated carefully after templating.

## Output-Length Modes

### Natural Response Mode

The model may stop at normal EOS. The benchmark logs actual output tokens and reports realistic user-facing latency. Throughput must be interpreted with actual output length in mind.

### Controlled Serving Mode

The benchmark uses an engine-supported method to produce approximately or exactly 128 output tokens where possible. This mode is synthetic performance testing for comparable TTFT, ITL, and throughput. It must be labeled separately from natural response mode.

Do not compare throughput unfairly when models emit very different output lengths.

## Runtime Strategy

SGLang is the first Sarvam-specific smoke-test runtime because the public NVIDIA/Sarvam study used SGLang.

For the fair cross-model benchmark, the priority is finding a common runtime across all included models. vLLM is an early common-runtime candidate because current vLLM documentation/registry coverage includes BharatGen Param2 and registry entries for Sarvam-30B/BharatGen Param2. Krutrim compatibility remains unknown until direct testing.

Decision rule:

1. Test Sarvam-30B with SGLang as a Sarvam-specific smoke test.
2. Test whether all included models work under one common runtime, with vLLM and SGLang both candidates.
3. Use one common runtime only for direct cross-model tables.
4. If a model cannot run under the chosen common runtime, exclude it from the core table and document why.
5. A Sarvam-only SGLang appendix is allowed later, clearly separated from cross-model results.

## Streaming Metrics

Primary metrics:

- TTFT: time from request send until the first generated content token is received.
- ITL/TPOT: latency between generated content tokens after the first.
- End-to-end latency: request send until final completion.
- Output tokens.
- Output tokens per second.
- Aggregate generated tokens per second under concurrency.
- Experiment-cell peak GPU memory.
- Success, failure, timeout, or OOM status.

## GPU Memory Methodology

Peak GPU memory is not a per-request metric during concurrency tests. It must be recorded at the experiment-cell level:

```text
model + runtime + workload bucket + language group + precision + concurrency
```

Per-request raw rows may reference the experiment-cell memory record, but should not pretend that the memory value belongs uniquely to one request.

## Result Schemas

Tokenizer raw JSONL rows must include:

- `experiment_id`
- `generated_at_utc`
- `model_id`
- `prompt_id`
- `parallel_group_id`
- `language`
- `script`
- `suite_type`
- `workload_type`
- `intent_category`
- `character_count`
- `formatted_input_tokens`
- `tokens_per_character`
- `template_success`
- `template_status`
- `tokenizer_load_success`
- `chat_template_present`
- `error_type`
- `error_message`

Future serving raw rows must include:

- `experiment_id`
- `timestamp_utc`
- `model_id`
- `model_revision`
- `runtime`
- `runtime_version`
- `hardware`
- `gpu_count`
- `precision_or_quantisation`
- `language`
- `suite_type`
- `workload_type`
- `target_input_token_bucket`
- `actual_formatted_input_tokens`
- `output_length_mode`
- `max_new_tokens`
- `actual_output_tokens`
- `concurrency`
- `repetition_id`
- `warmup_or_measured`
- `ttft_ms`
- `total_latency_ms`
- `itl_ms_or_tpot_ms`
- `output_tokens_per_second`
- `success`
- `error_type`
- `error_message`
- `generation_config`
- `prompt_id`
- `parallel_group_id`
- `gpu_memory_cell_id`

Future experiment-cell memory rows must include:

- `gpu_memory_cell_id`
- `experiment_id`
- `model_id`
- `runtime`
- `workload_type`
- `language`
- `precision_or_quantisation`
- `concurrency`
- `peak_gpu_memory_mb`
- `sampling_interval_ms`
- `measurement_notes`

Future aggregated rows must group by model, runtime, hardware, language, suite type, input-token bucket, output-length mode, and concurrency. They must include request counts, failure counts, P50/P95 TTFT, P50/P95 ITL or TPOT, P50/P95 total latency, mean output tokens/sec, aggregate generated throughput, completed requests/sec, peak observed VRAM, and relevant tokenizer-summary fields.

## Prefix-Cache Scenarios

Scenario 1: cold or minimally shared prompts.

Purpose: compare basic serving behavior without intentionally maximizing prefix reuse.

Scenario 2: shared-agent-prefix workload.

Purpose: represent many users served by the same agent/system prompt or tool schema. This scenario must define a shared prefix length, append language-specific final turns, and report results separately because prefix sharing can affect TTFT and throughput.

## Benchmark Stages

1. CPU-only tokenizer and schema validation.
2. Tiny smoke run.
3. Low-cost compatibility run.
4. Full single-request benchmark.
5. Limited concurrency benchmark.
6. Optional expansion only after go/no-go gates pass.

GPU serving code, launch scripts, and H100 execution are not part of the current milestone.
