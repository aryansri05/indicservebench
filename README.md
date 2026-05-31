# IndicServeBench

IndicServeBench is an experimental benchmark project for self-hosted streaming text-generation inference on Indian-language LLM workloads.

The project focuses on:

* Hindi in Devanagari
* Tamil in Tamil script
* Hinglish / code-mixed prompts in Latin script
* English as a reference-control language

The goal is to study practical inference behavior across Indian-language prompts, especially tokenization, time to first token, latency distribution, throughput, and deployment tradeoffs.

---

## Current Status

H100 SGLang pilot completed for Sarvam 30B FP8, with a preliminary same-H100 comparison against Qwen.

The current benchmark includes:

* Sarvam 30B FP8 measured on a single H100 SXM 80GB setup
* 240 measured Sarvam requests across English, Hindi, Tamil, and Hinglish/code-mixed prompts
* TTFT, mean latency, P90/P95 latency, throughput, input/output token tracking, and tokenization analysis
* Preliminary same-H100 Sarvam vs Qwen comparison
* Clear limitations around external setup, single concurrency, short prompts, no CUDA Graphs, and no hosted Sarvam API comparison

T4 and Apple M2 baselines are being expanded with measured raw runs.

---

## Important Scope Note

This benchmark is an external experimental study.

It is **not**:

* Sarvam’s internal production serving stack
* a hosted Sarvam API latency claim
* a reproduction of Sarvam/NVIDIA production performance
* a definitive model-quality ranking
* a claim that one model is universally better than another

The benchmark should be interpreted as a transparent external pilot study under a specific hardware/runtime setup.

---

## Research Question

Under controlled self-hosted streaming-serving conditions, how do Indian-language LLM workloads behave across:

* tokenizer cost
* time to first token
* inter-token / generation latency
* end-to-end latency
* generated-token throughput
* tail latency
* hardware cost tradeoffs
* stability under future concurrency tests

The long-term goal is to understand how language, tokenizer behavior, serving runtime, and hardware interact during practical LLM deployment.

---

## Current Pilot Setup

| Component                      | Details                                    |
| ------------------------------ | ------------------------------------------ |
| Primary model                  | Sarvam 30B FP8                             |
| Runtime                        | SGLang                                     |
| Hardware                       | Single H100 SXM 80GB                       |
| Environment                    | External RunPod-style setup                |
| Prompt suite                   | 48 prompts                                 |
| Language groups                | English, Hindi, Tamil, Hinglish/code-mixed |
| Prompts per language           | 12                                         |
| Passes                         | 5                                          |
| Total measured Sarvam requests | 240                                        |
| Concurrency                    | 1 request at a time                        |
| CUDA Graphs                    | Disabled                                   |
| Workload type                  | Short conversational prompts               |

---

## Metrics Tracked

The benchmark tracks:

* Time to First Token
* Mean latency
* P90 latency
* P95 latency
* Output tokens per second
* Input tokens
* Output tokens
* Raw prompt token counts
* Formatted chat-template token counts
* Tokenization overhead
* Language-wise tokenizer behavior
* Hardware deployment tradeoffs

---

## Why Tokenization Matters

Latency numbers are incomplete without token counts.

Two prompts that look similar to humans may have very different token counts depending on the tokenizer and language. This matters especially for Indian-language inference, where Hindi, Tamil, and code-mixed text may tokenize differently from English.

Input tokens mainly affect prompt processing / prefill behavior.

Output tokens mainly affect decode time and total generation latency.

Because of this, IndicServeBench records both raw prompt tokens and formatted chat-template tokens wherever possible.

---

## Sarvam 30B FP8 H100 Pilot

The Sarvam 30B FP8 pilot measured 240 requests on a single H100 SXM 80GB setup using SGLang.

The measured prompt set covered:

* 12 English prompts
* 12 Hindi prompts
* 12 Tamil prompts
* 12 Hinglish/code-mixed prompts

Each prompt was measured across 5 passes.

The benchmark tracked TTFT, latency distribution, tokenization behavior, and throughput for the selected workload.

---

## Preliminary Sarvam vs Qwen Observation

In a preliminary same-H100 SGLang streaming pilot, Sarvam 30B FP8 showed stable and lower median TTFT across English, Hindi, Tamil, and Hinglish/code-mixed prompts.

Qwen tokenized Hindi and Tamil prompts much more heavily and was slower overall in this specific setup.

| Language | Sarvam Median TTFT | Qwen Median TTFT | Sarvam Avg Raw Tokens | Qwen Avg Raw Tokens |
| -------- | -----------------: | ---------------: | --------------------: | ------------------: |
| English  |             ~62 ms |          ~115 ms |                 14.42 |               14.75 |
| Hindi    |             ~63 ms |          ~116 ms |                 17.75 |               66.67 |
| Tamil    |             ~63 ms |          ~145 ms |                 17.83 |               81.00 |
| Hinglish |             ~62 ms |          ~116 ms |                 17.00 |               22.33 |

### Interpretation

Sarvam’s TTFT stayed almost flat across all four language groups, even when moving from English to Hindi, Tamil, and Hinglish/code-mixed prompts.

Qwen used significantly more tokens for Hindi and Tamil, and Tamil showed a noticeable TTFT increase in this setup.

### Caveat

This supports tokenizer efficiency as an important signal for Indic-language inference, but it does **not** prove that tokenization alone caused the latency difference.

Runtime configuration, kernel selection, scheduler behavior, CUDA Graph availability, fallback kernels, and serving framework behavior can also affect TTFT and latency.

The Qwen comparison should be interpreted cautiously because the run required fallback FP8/MoE behavior after DeepGEMM/CUDA Graph issues, so it may not represent Qwen’s most optimized serving configuration.

---

## Final Observations

| Area              | Observation                                                                                       | Why it matters                                                                            |
| ----------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Latency stability | Sarvam’s median TTFT stayed stable across English, Hindi, Tamil, and Hinglish in the measured run | Suggests stable single-stream behavior across the pilot language set                      |
| Tokenization      | Hindi and Tamil tokenization differed significantly across models                                 | Tokenizer efficiency can affect latency, cost, and deployment behavior                    |
| Tail latency      | Mean, P90, and P95 latency were tracked instead of only average latency                           | Tail latency is more relevant for production-style serving than mean latency alone        |
| Prompt formatting | Formatted chat prompts can use more tokens than raw user prompts                                  | Real serving cost should account for full chat-template input, not only visible user text |
| Hardware tradeoff | H100 is powerful but single-stream cost can look high without batching/concurrency                | Production serving requires utilization, batching, and scheduling analysis                |
| Runtime scope     | Results depend on the external SGLang setup used for the pilot                                    | The benchmark should not be treated as Sarvam’s internal production performance           |
| Future direction  | Concurrency, TTFT breakdown, CUDA Graphs, and profiling remain future work                        | These are required for a more production-relevant benchmark                               |

Overall, the benchmark shows that latency numbers alone are incomplete without token counts, prompt formatting details, tail-latency metrics, runtime configuration, and hardware-utilization context.

---

## Benchmark Limitations

This benchmark should be treated as an external experimental study, not a claim about Sarvam’s internal production performance.

| Limitation                    | Details                                                                                                                                                                                       |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Not Sarvam’s production stack | The benchmark was run on an external RunPod/SGLang setup, not Sarvam’s internal production serving infrastructure.                                                                            |
| Single H100 setup             | The main measured run used one H100 SXM 80GB GPU, not a multi-GPU production deployment.                                                                                                      |
| Single concurrency            | Requests were sent one at a time. The benchmark does not evaluate batching, real traffic load, queueing behavior, or mixed prefill/decode scheduling under concurrency.                       |
| Small prompt suite            | The prompt suite contained 48 prompts: 12 English, 12 Hindi, 12 Tamil, and 12 Hinglish/code-mixed prompts. This is useful for a pilot benchmark, but not enough for broad statistical claims. |
| Short-context workload        | Prompts were mostly short conversational inputs. Long-context workloads were not tested.                                                                                                      |
| Client-side timing            | Latency was measured from the benchmark client. The current setup does not fully isolate tokenizer time, queue time, prefill time, decode time, network overhead, and server-side scheduling. |
| Runtime not fully optimized   | CUDA Graphs were disabled, so the benchmark should be treated as a conservative baseline rather than a fully production-tuned serving run.                                                    |
| Qwen comparison caveat        | The Qwen comparison required fallback FP8/MoE behavior after DeepGEMM/CUDA Graph issues, so Qwen was not running in its most optimized configuration.                                         |
| No hosted API comparison      | The benchmark does not compare against Sarvam’s hosted API latency, so it should not be interpreted as a product latency claim.                                                               |

These limitations are intentional. The goal was to create a transparent baseline benchmark and identify future directions such as concurrency testing, TTFT breakdown, CUDA Graphs, server-side profiling, and larger Indic prompt suites.

---

## Planned Next Steps

Future work includes:

* Add concurrency testing
* Measure TTFT, inter-token latency, queue time, prefill time, and decode time separately
* Add CUDA Graphs comparison
* Add Nsight / SGLang profiling
* Expand prompt suite beyond 48 prompts
* Test longer-context workloads
* Replace T4 and Apple M2 baselines with fully measured raw runs
* Compare vLLM and SGLang where feasible
* Add larger Indic-language prompt suites
* Add cost-per-1M-token estimates under batching/concurrency

---

## Candidate Models

The broader v1 candidate model list includes:

* `sarvamai/sarvam-30b`
* `krutrim-ai-labs/Krutrim-2-instruct`
* `bharatgenai/Param2-17B-A2.4B-Thinking`

Additional pilot comparisons may include Qwen-family models where useful as a reference baseline.

These models are not architecturally identical. Some models may be dense, while others may use MoE-style architectures. Any comparison should be framed as observed serving behavior under the same test conditions, not as proof that one architecture or model is generally superior.

---

## Public Workload Inspiration

The long-context workload direction is inspired by a public NVIDIA/Sarvam engineering write-up discussing Sarvam 30B inference optimization for real-time voice-agent-style workloads using SGLang, H100-class hardware, P95 TTFT, P95 ITL, an average input length of 3,584 tokens, and 128 output tokens.

Any H100 results in this repository should be described as externally inspired by the reported workload shape. They should not be described as a reproduction of Sarvam/NVIDIA production performance.

Source:

https://developer.nvidia.com/blog/how-nvidia-extreme-hardware-software-co-design-delivered-a-large-inference-boost-for-sarvam-ais-sovereign-models/

---

## Benchmark Tracks

1. Documentation and feasibility specification
2. Tokenizer analysis
3. Runtime compatibility smoke testing
4. Single-request streaming latency
5. Concurrent streaming serving
6. Published cross-model report
7. Optional T4-versus-H100 experiment when the same configuration is feasible on both GPUs
8. Optional Apple Silicon local baseline for practical comparison

---

## Runtime Strategy

SGLang remains the first Sarvam-specific runtime because the public NVIDIA/Sarvam study used SGLang.

For direct cross-model comparison, the priority is to use one common runtime where feasible.

vLLM is an important common-runtime candidate because of its broad model support and production relevance. However, final direct-comparison tables should clearly state the runtime used and avoid mixing results from different runtimes without caveats.

Runtime compatibility is not assumed until smoke-tested.

---

## Historical Pre-GPU Milestone

The project originally started with a pre-GPU milestone that created:

* documentation
* prompt schemas
* prototype natural prompts
* model configuration metadata
* tokenizer-analysis tooling
* prompt-schema tests

That milestone did not launch SGLang, vLLM, or any H100 instance. It has now been followed by the H100 SGLang pilot described above.

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

Actual tokenizer counts require a Python environment with `transformers` and `jinja2` available.

If `transformers`, a tokenizer, or a chat template cannot load, the tokenizer-analysis script records structured failure rows instead of crashing the whole run.

The tokenizer diagnostic writes each run to:

```text
results/tokenizer/<experiment_id>/
```

with:

```text
raw.jsonl
summary.csv
metadata.json
```

It records raw user-prompt token counts, formatted chat-template token counts, and template overhead.

---

## Repository Goals

IndicServeBench aims to be:

* transparent
* reproducible
* careful about caveats
* useful for Indian-language inference analysis
* honest about what has and has not been measured
* focused on serving behavior, not model-quality ranking

The project is still evolving. The current H100 results should be treated as a pilot benchmark, not a final benchmark suite.
