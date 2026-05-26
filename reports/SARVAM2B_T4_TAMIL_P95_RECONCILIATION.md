# Sarvam-2B T4 Tamil P95 Reconciliation

This note reconciles the earlier Sarvam-2B T4 benchmark against the original
prompt lengths that could be recovered from the saved local artifacts. It is a
CPU-only tokenizer analysis. It does not run GPU serving code, and it does not
claim a CUDA/kernel bottleneck, a Sarvam-30B result, lower serving latency, or
model quality.

## Artifacts Checked

Found:

- `/Users/aryansrivastava/Downloads/sarvam_baseline.json`
- `/Users/aryansrivastava/Downloads/sarvam_baseline (1).json`
- `/Users/aryansrivastava/Downloads/sarvam benchmarks collab (1).ipynb`
- `/Users/aryansrivastava/Downloads/sarvam benchmarks collab.ipynb`
- `/Users/aryansrivastava/Downloads/sarvam_benchmark_t4.ipynb`

The notebook contains the exact 10 Hindi, 10 Tamil, and 10 code-mixed prompts
used by the baseline batch. The downloaded JSON contains only aggregate
language-level results.

Unavailable:

- Saved per-request latency rows for all 30 prompts.
- Saved per-request generated output token counts.
- Saved generated response text for the batch run.
- Prompt IDs joined to latency values.

The notebook's `run_batch` function did create in-memory rows with `lang`,
`ms`, generated `tokens`, and `tps`, but the downloaded JSON did not preserve
those rows. Therefore, exact Tamil tail-latency outliers cannot be identified
from the available saved artifacts.

## Method

Tokenizer loaded CPU-only:

- `sarvamai/sarvam-2b-v0.5`
- `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`

No model weights were loaded.

The original benchmark wrapped each prompt as:

```text
<s>[INST] {prompt} [/INST]
```

To match the benchmark accounting as closely as possible, input token counts
below use the same default tokenizer call style as the notebook:

```text
tokenizer("<s>[INST] {prompt} [/INST]")
```

The old batch used `max_new_tokens=64`, `temperature=0.1`, and `do_sample=True`.
Because actual generated-token rows were not saved, output-token summaries are
unavailable.

One implementation detail matters for interpreting the reported P95: the
notebook computed `sorted(times)[int(0.95 * len(times))]`. With 10 prompts per
language, that selects index 9, which is the maximum observed latency, not an
interpolated statistical P95.

## Grouped Summary

| Language | Prompt count | Mean input tokens | Median input tokens | Min input tokens | Max input tokens | Mean output tokens | Median output tokens | Median latency ms | Reported P95/max latency ms |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| Hindi | 10 | 11.6 | 11.0 | 10 | 14 | unavailable | unavailable | 2471.2 | 3099.1 |
| Tamil | 10 | 12.5 | 12.5 | 11 | 15 | unavailable | unavailable | 2535.9 | 5477.0 |
| Code-mixed Hindi-English | 10 | 24.7 | 22.0 | 18 | 37 | unavailable | unavailable | 2468.5 | 2996.7 |

## Tamil Tail-Latency Outliers

Confirmed Tamil tail-latency outliers cannot be listed because the saved
artifacts do not contain per-request latency rows. The table below lists the
longest Tamil prompts by benchmark input tokens only. These are token-length
candidates, not confirmed latency outliers.

| Tamil prompt index | Benchmark input tokens | Raw prompt tokens | Characters | Prompt |
|---:|---:|---:|---:|---|
| 2 | 15 | 10 | 39 | தமிழ்நாட்டின் வரலாறு பற்றி சொல்லுங்கள். |
| 7 | 14 | 9 | 34 | தமிழ் இலக்கியம் பற்றி சொல்லுங்கள். |
| 3 | 13 | 8 | 41 | நல்ல உணவு பழக்கம் எப்படி இருக்க வேண்டும்? |
| 5 | 13 | 8 | 43 | கணினி பயன்படுத்துவது எப்படி கற்றுக்கொள்வது? |
| 6 | 13 | 8 | 29 | வங்கி கணக்கு திறப்பது எப்படி? |

For comparison, the longest Hindi prompt had 14 benchmark input tokens, and the
Hindi median was 11 tokens. Code-mixed prompts were substantially longer overall
with a 24.7-token mean and 37-token maximum, yet their reported P95/max latency
was 2996.7 ms rather than a Tamil-like 5477.0 ms.

## Reconciliation

The available input-token evidence does not strongly support the idea that the
Tamil P95/max-latency spike was simply caused by longer input prompts. Tamil
inputs were only slightly longer than Hindi inputs:

- Tamil mean input tokens: 12.5
- Hindi mean input tokens: 11.6
- Tamil max input tokens: 15
- Hindi max input tokens: 14

Meanwhile, code-mixed prompts were much longer by input-token count but did not
show the same tail spike in the saved aggregate summary.

However, causation cannot be resolved from the saved aggregate JSON. The missing
per-request rows prevent checking whether the slow Tamil request had:

- unusually many generated output tokens;
- an early or late EOS;
- an anomalous generation path from sampling;
- a one-off runtime hiccup;
- a specific prompt-dependent behavior.

## Conclusion

Based on recovered prompt lengths alone, the Tamil tail-latency spike is not
well explained by input-token length differences. The strongest defensible
statement is:

> The saved artifacts show only a small Tamil-vs-Hindi input-token difference,
> while code-mixed prompts were much longer without the same reported tail.
> Therefore, input prompt length alone is an unlikely complete explanation for
> the Tamil P95/max spike, but the original raw per-request latency and output
> token data are missing, so no causal explanation can be established.

This old Sarvam-2B T4 result and the newer Sarvam-30B/Qwen tokenizer diagnostic
are related motivation only. They are not directly comparable evidence.

## Planned Controlled Reproduction

The prior Tamil P95/max observation is treated as motivation only. A controlled
Sarvam-2B/T4 reproduction runner has been prepared to rerun the exact recovered
Hindi, Tamil, and code-mixed prompts while preserving request-level latency,
input-token counts, output-token counts, prompt IDs, GPU metadata, and failure
status.

The reproduction will use warmup rows, shuffled measured request order, and
linear-interpolated P90/P95 aggregation rather than repeating the old
max-as-P95 summary behavior. No attention-layer or CUDA/kernel bottleneck is
currently claimed.
