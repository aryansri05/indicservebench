# Experiment Log Template

Use this template for every run or session, including failed smoke tests. No benchmark results exist yet.

## Run Identity

- Experiment ID:
- Run ID:
- Date:
- Operator:
- Purpose:
- Gate approved:
- Warmup or measured:

## Hardware

- Provider:
- Region:
- Instance type:
- GPU model:
- GPU count:
- GPU memory:
- CPU:
- RAM:
- Storage:
- Hourly rate:
- Credit status:

## Software

- OS image:
- Python version:
- CUDA version:
- NVIDIA driver version:
- Runtime:
- Runtime version:
- Transformers version:
- PyTorch version:
- Other relevant packages:

## Model

- Model ID:
- Model revision/commit:
- License checked:
- Context limit:
- Precision or quantization:
- `trust_remote_code`:
- Chat template status:

## Prompt And Generation

- Prompt file:
- Prompt IDs:
- Suite type:
- Workload bucket:
- Output-length mode:
- Max new tokens:
- Temperature:
- Top-p:
- Top-k:
- Sampling enabled:
- Seed:
- Stop-token behavior:

## Commands

Record exact commands used:

```text
TODO
```

## Results

- Raw result path:
- Summary path:
- Server log path:
- Token event trace path:
- Peak GPU memory cell record:

## Failures And Observations

- Failure type:
- Error message:
- OOM:
- Timeout:
- Malformed output:
- Context rejection:
- Notes:

## Shutdown Confirmation

- Artifacts retrieved:
- Processes stopped:
- GPU resource destroyed:
- Billing console checked:
- Budget ledger updated:
