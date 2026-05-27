# Sarvam H100 SGLang Five-Pass Interpretation

This is a preliminary single-H100 SGLang streaming pilot, not a production benchmark.

## Data accounting

The output folders contain five complete passes of the frozen 48-prompt suite. Due to rerunning with the same experiment IDs, two passes were appended into `sarvam_h100_sglang_pilot_v2` and two passes were appended into `sarvam_h100_sglang_pilot_v3`.

| output_folder               |   complete_passes_stored |   measured_requests |
|:----------------------------|-------------------------:|--------------------:|
| sarvam_h100_sglang_pilot_v1 |                        1 |                  48 |
| sarvam_h100_sglang_pilot_v2 |                        2 |                  96 |
| sarvam_h100_sglang_pilot_v3 |                        2 |                  96 |

Total measured requests: **240**  
Measured requests per language: **60** 

## Interpretation

Across five complete passes of the frozen 48-prompt suite, median TTFT remained close across English, Hindi, Tamil and Hinglish. Tamil did not show elevated TTFT. Four transient tail events above 150 ms occurred in the initial output folder only—three Hindi prompts and one Hinglish prompt—and did not provide repeatable evidence of a language-specific or tokenizer-driven latency issue.

## Summary by language

| language   |   n |   median_ttft_ms |   p90_ttft_ms |   max_ttft_ms |   median_total_latency_ms |   mean_output_tps |   tail_events_over_150ms |
|:-----------|----:|-----------------:|--------------:|--------------:|--------------------------:|------------------:|-------------------------:|
| en         |  60 |          62.054  |       63.5157 |        70.36  |                   879.053 |           36.2497 |                        0 |
| hi         |  60 |          62.9095 |       71.551  |       536.909 |                   879.975 |           35.6534 |                        3 |
| hinglish   |  60 |          62.4145 |       70.5771 |       494.496 |                   881.333 |           36.0303 |                        1 |
| ta         |  60 |          62.9905 |       67.201  |        74.818 |                   880.426 |           36.236  |                        0 |

## Tail events above 150 ms

| source_folder               | language   | prompt_id        |   ttft_ms |   total_latency_ms |   end_to_end_output_tokens_per_second |
|:----------------------------|:-----------|:-----------------|----------:|-------------------:|--------------------------------------:|
| sarvam_h100_sglang_pilot_v1 | hi         | nat_002_hi       |   536.909 |            1349.72 |                                23.709 |
| sarvam_h100_sglang_pilot_v1 | hinglish   | nat_008_hinglish |   494.496 |            1327.69 |                                24.102 |
| sarvam_h100_sglang_pilot_v1 | hi         | nat_009_hi       |   485.342 |            1308.41 |                                24.457 |
| sarvam_h100_sglang_pilot_v1 | hi         | nat_005_hi       |   467.173 |            1305.59 |                                24.51  |

## Claim boundary

This result is exploratory and based on a single-H100 external SGLang configuration. It is not Sarvam's published tuned multi-GPU configuration, not a production benchmark, and not evidence of a CUDA/kernel bottleneck.
