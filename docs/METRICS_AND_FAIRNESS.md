# Metrics And Fairness

Current status: metric specification only. No benchmark results exist yet.

## Metric Definitions

TTFT is the elapsed time from sending the request until receiving the first generated content token. Metadata-only chunks do not count as generated content tokens.

ITL is the latency between consecutive generated content tokens after the first token.

TPOT is a compact per-request summary of inter-token behavior:

```text
(timestamp_last_output_token - timestamp_first_output_token) / (output_tokens - 1)
```

TPOT is undefined when fewer than two output tokens are produced.

End-to-end latency is the elapsed time from sending the request until receiving the final completion event.

Output tokens per second is based on generated output tokens, not input tokens.

Aggregate generated throughput is total generated output tokens per second across all completed measured requests in an experiment cell.

Peak GPU memory during concurrency tests is an experiment-cell metric, not a per-request metric.

## Output-Length Methodology

Natural response mode allows normal EOS and logs actual output tokens. This mode represents realistic user-facing latency.

Controlled serving mode uses an engine-supported method to produce approximately or exactly 128 output tokens where possible. This mode is synthetic and supports fairer TTFT, ITL, and throughput comparison.

Reports must not compare throughput unfairly when models emit very different output lengths.

## Fair Direct Comparison Rules

Direct comparison tables require:

- Same model revision policy.
- One common runtime.
- Same runtime version.
- Same generation settings.
- Same prompt suite.
- Same output-length mode.
- Same formatted input-token bucket.
- Same measurement procedure.
- Compatible precision or quantization policy.
- Same hardware class and GPU count.

Do not mix SGLang and vLLM rows into one direct model leaderboard. Do not create T4-versus-H100 speed tables unless the exact same model checkpoint, precision/quantization format, serving engine, generation settings, prompt suite, formatted token profile, and measurement procedure are used on both GPUs.

## Valid Claims

Valid conclusions:

- Model A had lower observed P95 TTFT than Model B under a specified common runtime, hardware, prompt suite, output mode, and precision.
- Tamil prompts produced more formatted input tokens than Hinglish for a specified tokenizer and prompt suite.
- A model was excluded from the core benchmark because it failed a documented common-runtime smoke test.
- A shared-prefix workload changed observed TTFT or throughput relative to a minimally shared workload.

Overclaims:

- One model is generally faster in all deployments.
- MoE is generally superior to dense architecture.
- Results reproduce Sarvam/NVIDIA production performance.
- Tokenizer differences alone explain all latency differences.
- A model is better quality because it is faster.

## Failure Reporting

Failed requests are data. Logs must preserve:

- Failure type.
- Error message.
- Runtime.
- Prompt ID.
- Context-token count.
- Output cap.
- Concurrency level.
- Whether the request was warmup or measured.

OOMs, timeouts, context-limit rejections, malformed streaming responses, and server crashes must not be silently discarded.
