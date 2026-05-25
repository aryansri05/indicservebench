# H100 Safety Checklist

Current status: safety plan only. Do not launch H100 resources for the current milestone.

## Rule Zero

No H100 should be launched until cloud credit is visible and usable, billing alerts are configured, and the pre-GPU gate is complete.

## Pre-Launch Checklist

- Project specification approved.
- Model identifiers verified from official pages.
- Prompt schema finalized.
- Raw result schema finalized.
- Basic tokenizer analysis completed or ready to run.
- Benchmark-client design completed.
- Server-launch commands prepared conceptually.
- Automatic saving/logging plan prepared.
- Cloud credit visible.
- Billing alert configured.
- Manual timer prepared.
- Shutdown and destroy checklist prepared.
- Result retrieval path prepared.

## First H100 Session Scope

The first H100 session should be capped at approximately one hour.

Allowed first-session scope:

- One model: `sarvamai/sarvam-30b`.
- One runtime: SGLang.
- A few short prompts only.
- Streaming response validation.
- Basic TTFT capture.
- Maximum output 32 or 64 tokens.

Not allowed in the first session:

- Cross-model benchmark.
- Long-context benchmark.
- Concurrency sweep.
- Performance chart production.
- Any claim of final benchmark results.

## Running-Session Checklist

- Start external timer before provisioning.
- Record instance type, region, hourly cost, and start time.
- Confirm GPU visibility.
- Save server logs continuously.
- Save client results incrementally.
- Stop immediately on repeated OOM, server crash, billing uncertainty, or unexpected model download behavior.

## Shutdown And Destroy Checklist

- Save raw results.
- Save server logs.
- Save environment/version information.
- Retrieve artifacts from the instance.
- Stop all benchmark processes.
- Destroy the GPU resource. Do not merely power it off.
- Confirm in the cloud console that the billable resource is destroyed.
- Update the budget ledger.

## Budget Ledger Template

| Date | Provider | Instance Type | Hourly Rate | Start Time | End Time | Duration Hours | Estimated Cost | Credit Before | Credit After | Artifacts Saved | Resource Destroyed | Next Gate |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |

## Go/No-Go Principle

No full experiment should run until a short smoke test works. Every expensive phase needs a written go/no-go decision.
