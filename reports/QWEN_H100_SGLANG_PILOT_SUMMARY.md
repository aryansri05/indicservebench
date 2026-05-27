# Qwen H100 SGLang Streaming Pilot

This is a preliminary H100 SGLang streaming pilot, single concurrency, using the frozen natural prompt suite. It is not a production benchmark and does not establish a CUDA/kernel bottleneck or a causal tokenizer-latency claim.

- Model: `qwen` (`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`)
- Run ID: `qwen_h100_sglang_pilot_20260527T105809Z`

| Language | n | successes | median TTFT ms | p90 TTFT ms | median total ms | mean raw toks | mean formatted toks | mean output toks | mean tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 12 | 12 | 115.496 | 118.145 | 1826.478 | 14.75 | 57.833 | 32.0 | 17.515 |
| hi | 12 | 12 | 116.442 | 117.963 | 1829.059 | 66.667 | 214.5 | 32.167 | 17.573 |
| ta | 12 | 12 | 144.982 | 170.918 | 1829.736 | 81.0 | 250.167 | 32.167 | 17.567 |
| hinglish | 12 | 12 | 115.62 | 117.807 | 1826.847 | 22.333 | 73.083 | 32.0 | 17.501 |

Meeting wording: preliminary Qwen3-30B-A3B-FP8 H100 SGLang streaming pilot, single concurrency, frozen 48-prompt natural suite. Do not describe this as production performance.
