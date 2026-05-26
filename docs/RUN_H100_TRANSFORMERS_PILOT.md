# Run H100 Transformers Pilot

Purpose: produce a small, defensible preliminary result for the frozen 48-prompt natural suite on one Runpod H100 SXM, using only Hugging Face Transformers streaming generation.

Strict claim boundary: this is `preliminary_h100_transformers_streaming_pilot`. It is not a production benchmark, not an SGLang/vLLM benchmark, not a Sarvam optimization study, not a proven tokenizer bottleneck, and not a full serving result. TTFT here means Python client-observed time from starting `model.generate(...)` to the first non-empty generated text chunk observed from `TextIteratorStreamer`.

## Why Transformers Streaming

Tonight's goal is to compare Sarvam and, optionally, Qwen under the same single-H100, same Python/Transformers path. The public Sarvam optimization work used serving-runtime and kernel-level changes, but this pilot intentionally does not continue SGLang/vLLM/CUDA debugging. If Sarvam FP8 does not run under basic Transformers, preserve the artifacts and report that result honestly.

Official model references:

- Sarvam FP8: <https://huggingface.co/sarvamai/sarvam-30b-fp8>
- Qwen FP8: <https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8>

## 1. H100 Preflight

In Runpod Jupyter, open **Terminal** from the launcher. Paste:

```bash
cd /workspace
export HF_HOME=/workspace/huggingface
export HF_HUB_CACHE=/workspace/huggingface/hub
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" /workspace/indicservebench_results

nvidia-smi
df -h /workspace
python3 - <<'PY'
import torch
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if torch.cuda.is_available() else "None")
PY
```

Stop if the GPU name does not contain `H100`.

## 2. Get The Latest Runner

If the repo is already cloned:

```bash
cd /workspace/indicservebench
git pull --ff-only
git log --oneline -5
```

If it is not cloned:

```bash
cd /workspace
git clone https://github.com/aryansri05/indicservebench.git
cd /workspace/indicservebench
git log --oneline -5
```

## 3. Install Minimal Dependencies

Preserve the pod's working CUDA/PyTorch install. Do not install SGLang or vLLM.

```bash
python3 -m pip install -U \
  "transformers==4.51.3" \
  "accelerate" \
  "safetensors" \
  "sentencepiece" \
  "protobuf<7" \
  "pandas" \
  "pytest"

# Avoid a known incompatible torchvision import path in some Runpod images.
python3 -m pip uninstall -y torchvision || true

python3 - <<'PY'
import torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("imports_ok")
PY
```

## 4. Run CPU Tests On The Pod

```bash
cd /workspace/indicservebench
PYTHONPATH=src python3 -m pytest tests/test_h100_transformers_streaming_pilot.py -q
```

## 5. Sarvam Smoke Test

Run Sarvam first. This loads only `sarvamai/sarvam-30b-fp8`, runs one warmup request and one measured request per language, and writes request-level JSONL immediately.

```bash
cd /workspace/indicservebench
export HF_HOME=/workspace/huggingface
export HF_HUB_CACHE=/workspace/huggingface/hub

EXP=sarvam_h100_smoke_$(date -u +%Y%m%dT%H%M%SZ)
PYTHONPATH=src python3 -m indicservebench.h100_transformers_streaming_pilot \
  --smoke \
  --model sarvam \
  --experiment-id "$EXP" \
  --output-root /workspace/indicservebench_results/h100_transformers_pilot \
  --workspace-root /workspace \
  --device-map cuda

cat /workspace/indicservebench_results/h100_transformers_pilot/$EXP/human_readable_summary.md
```

Inspect raw rows, especially if the command exits non-zero:

```bash
python3 - <<'PY'
import json, os
exp=os.environ["EXP"]
p=f"/workspace/indicservebench_results/h100_transformers_pilot/{exp}/raw_requests.jsonl"
for line in open(p, encoding="utf-8"):
    r=json.loads(line)
    print(r["warmup_or_measured"], r["prompt_id"], r["language"], r["success"], r["ttft_ms"], r["error_type"], r["error_message"])
PY
```

Stop if Sarvam fails to load, produces empty/corrupted output, hits CUDA OOM, or shows an unsupported FP8/runtime error such as `expected mat1 and mat2 to have the same dtype`. Preserve the output directory and do not pivot to SGLang tonight.

## 6. Sarvam 48-Prompt Pilot

Only run this if the Sarvam smoke test succeeds with non-empty outputs.

```bash
cd /workspace/indicservebench
export HF_HOME=/workspace/huggingface
export HF_HUB_CACHE=/workspace/huggingface/hub

EXP=sarvam_h100_pilot_$(date -u +%Y%m%dT%H%M%SZ)
PYTHONPATH=src python3 -m indicservebench.h100_transformers_streaming_pilot \
  --pilot \
  --model sarvam \
  --experiment-id "$EXP" \
  --output-root /workspace/indicservebench_results/h100_transformers_pilot \
  --workspace-root /workspace \
  --device-map cuda

cat /workspace/indicservebench_results/h100_transformers_pilot/$EXP/human_readable_summary.md
```

Save the Sarvam artifacts immediately:

```bash
cd /workspace
zip -r sarvam_h100_pilot_results.zip indicservebench_results/h100_transformers_pilot/$EXP
ls -lh /workspace/sarvam_h100_pilot_results.zip
df -h /workspace
```

## 7. Optional Qwen Smoke And Pilot

Run Qwen only after Sarvam succeeds, artifacts are saved, time remains, and `/workspace` has enough free disk.

```bash
cd /workspace/indicservebench
export HF_HOME=/workspace/huggingface
export HF_HUB_CACHE=/workspace/huggingface/hub

QWEN_SMOKE=qwen_h100_smoke_$(date -u +%Y%m%dT%H%M%SZ)
PYTHONPATH=src python3 -m indicservebench.h100_transformers_streaming_pilot \
  --smoke \
  --model qwen \
  --experiment-id "$QWEN_SMOKE" \
  --output-root /workspace/indicservebench_results/h100_transformers_pilot \
  --workspace-root /workspace \
  --device-map cuda

cat /workspace/indicservebench_results/h100_transformers_pilot/$QWEN_SMOKE/human_readable_summary.md
```

If Qwen smoke succeeds:

```bash
QWEN_EXP=qwen_h100_pilot_$(date -u +%Y%m%dT%H%M%SZ)
PYTHONPATH=src python3 -m indicservebench.h100_transformers_streaming_pilot \
  --pilot \
  --model qwen \
  --experiment-id "$QWEN_EXP" \
  --output-root /workspace/indicservebench_results/h100_transformers_pilot \
  --workspace-root /workspace \
  --device-map cuda

cat /workspace/indicservebench_results/h100_transformers_pilot/$QWEN_EXP/human_readable_summary.md
```

## 8. Comparison Summary

Only run this after both 48-prompt pilots complete successfully.

```bash
COMPARE_EXP=sarvam_qwen_h100_comparison_$(date -u +%Y%m%dT%H%M%SZ)
PYTHONPATH=src python3 -m indicservebench.h100_transformers_streaming_pilot \
  --compare \
  --experiment-id "$COMPARE_EXP" \
  --output-root /workspace/indicservebench_results/h100_transformers_pilot \
  --sarvam-run-dir /workspace/indicservebench_results/h100_transformers_pilot/$EXP \
  --qwen-run-dir /workspace/indicservebench_results/h100_transformers_pilot/$QWEN_EXP

cat /workspace/indicservebench_results/h100_transformers_pilot/$COMPARE_EXP/comparison_summary.md
```

## 9. Zip And Download

```bash
cd /workspace
zip -r indicservebench_h100_transformers_pilot_outputs.zip indicservebench_results/h100_transformers_pilot
ls -lh /workspace/indicservebench_h100_transformers_pilot_outputs.zip
```

Download the zip from the Jupyter file browser, then terminate the Runpod pod immediately.

## Stop Conditions

Stop and preserve logs if any of these happen:

- GPU is not H100.
- Sarvam model load fails.
- Output is empty or corrupted.
- CUDA out-of-memory occurs.
- Setup/debugging exceeds 45 minutes before first Sarvam successful output.
- Sarvam does not produce a saved result within 60 minutes.
- Any unsupported model/runtime error appears.

Do not pivot to SGLang tonight.

## Meeting Interpretation Wording

Use only cautious wording:

- "This was a preliminary single-process Transformers streaming pilot on one H100."
- "TTFT is Python client-observed streamer TTFT, not production server TTFT."
- "Sarvam showed lower/higher median TTFT than Qwen for this language in this pilot."
- "This is consistent/inconsistent with the tokenizer-count hypothesis, but does not establish causality."
- "Output-token counts and formatted input-token counts must be checked before attributing latency differences to language or serving behavior."
- "No CUDA/kernel bottleneck is claimed."
- "A production-serving conclusion requires a serving-runtime study such as SGLang/vLLM under a validated supported configuration."
