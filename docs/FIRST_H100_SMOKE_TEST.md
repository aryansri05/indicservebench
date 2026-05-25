# First H100 Smoke Test

Current status: preparation only. Do not run this smoke test until the H100
go/no-go gate is explicitly approved, cloud credit is visible, and billing
alerts are active.

## Purpose

Validate that `sarvamai/sarvam-30b-fp8` streams on one H100 and that the
OpenAI-compatible streaming client records TTFT correctly.

This is not a benchmark run and must not be reported as a performance result.
It is a minimal serving smoke test.

## Fixed Scope

- Model to test: `sarvamai/sarvam-30b-fp8`
- GPU: one H100 only
- Prompt count: three existing short natural prompts only
- Languages: one Hindi, one Tamil, one Hinglish prompt
- Output cap: 32 tokens
- Concurrency: 1
- Maximum session duration: 60 minutes
- Runtime target: SGLang, pending final one-GPU FP8 launch verification

Do not run concurrency, long-context prompts, cross-model comparisons, charts,
or additional model tests in this session.

## Billing And Credit Check

Before provisioning:

- Confirm cloud credit is visible and usable.
- Confirm billing alerts are configured.
- Confirm the selected H100 resource price and region.
- Start an external 60-minute timer before creating the GPU Droplet.
- Open the budget ledger before launch.

## SGLang Launch Command

Final verified one-GPU SGLang launch command:

```bash
# PENDING VERIFICATION.
# Do not use this placeholder as a launch command.
# The exact one-GPU command must be verified from the Sarvam FP8 model/runtime
# documentation or from a minimal launch test before the smoke session.
```

## Client Command

Once an OpenAI-compatible server is already running, execute the client from
the repository root:

```bash
source .venv/bin/activate
python src/indicservebench/streaming_smoke_client.py \
  --server-base-url "http://127.0.0.1:PORT" \
  --model-id "sarvamai/sarvam-30b-fp8" \
  --runtime-label "sglang" \
  --run-id "first_h100_sarvam_fp8_smoke"
```

The client sends exactly three streaming chat-completion requests with
`max_tokens=32` and writes each result immediately to:

```text
results/streaming_smoke/first_h100_sarvam_fp8_smoke/raw.jsonl
```

## Result-Copy Checklist

Before destroying the GPU resource:

- Copy `results/streaming_smoke/first_h100_sarvam_fp8_smoke/raw.jsonl`.
- Copy `results/streaming_smoke/first_h100_sarvam_fp8_smoke/metadata.json`.
- Copy the server log.
- Copy the exact launch command actually used.
- Copy package/runtime version information.
- Confirm copied files are readable locally.

## Stop Conditions

Stop the session immediately if:

- The model fails to launch after the minimal documented attempt.
- Streaming returns no content for all three prompts.
- TTFT logging fails.
- Billing or credit status is unclear.
- The session approaches 60 minutes.

## Destroy Resource

After copying results, destroy the DigitalOcean GPU Droplet. Do not merely
power it off. Confirm in the DigitalOcean console that the billable GPU Droplet
has been destroyed, then update the budget ledger.
