# Run Sarvam H100 SGLang Pilot

Purpose: get a small Sarvam-on-H100 result tonight. This is a preliminary SGLang streaming pilot, single concurrency, using the frozen natural 48-prompt suite. It is not production performance.

## 1. Keep Server Terminal Separate

Use two Jupyter terminal tabs:

- Tab 1: server and `tail -f` logs.
- Tab 2: client benchmark commands.

## 2. Fix Missing NUMA Library

Run this if `sgl_kernel` fails with `libnuma.so.1`:

```bash
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libnuma1 numactl
ldconfig

python3 - <<'PY'
import importlib.metadata as md
print("sglang-kernel:", md.version("sglang-kernel"))
import sgl_kernel
print("sgl_kernel import succeeded")
PY
```

## 3. Launch Sarvam Server

```bash
mkdir -p /workspace/sglang_logs
export PYTHONPATH=/workspace/sglang/python:$PYTHONPATH
export HF_HOME=/workspace/huggingface
export HF_HUB_CACHE=/workspace/huggingface/hub

nohup python3 -m sglang.launch_server \
  --model-path sarvamai/sarvam-30b-fp8 \
  --host 127.0.0.1 \
  --port 30000 \
  --trust-remote-code \
  --quantization modelopt_fp8 \
  --mem-fraction-static 0.70 \
  > /workspace/sglang_logs/sarvam_h100_server.log 2>&1 &

echo "PID: $!"
tail -n 100 -f /workspace/sglang_logs/sarvam_h100_server.log
```

When the server is ready, press `Ctrl-C` to stop tailing logs. This does not stop the server.

Check the server from the second terminal:

```bash
curl -s http://127.0.0.1:30000/v1/models | head
```

## 4. Pull Client Runner

```bash
cd /workspace/indicservebench
git pull --ff-only
git log --oneline -5

PYTHONPATH=src python3 -m pytest tests/test_h100_sglang_sarvam_pilot.py -q
```

## 5. Run Smoke

```bash
cd /workspace/indicservebench
export HF_HOME=/workspace/huggingface
export HF_HUB_CACHE=/workspace/huggingface/hub

EXP=sarvam_h100_sglang_smoke_$(date -u +%Y%m%dT%H%M%SZ)
PYTHONPATH=src python3 -m indicservebench.h100_sglang_sarvam_pilot \
  --smoke \
  --experiment-id "$EXP" \
  --base-url http://127.0.0.1:30000 \
  --output-root /workspace/indicservebench_results/sarvam_h100_sglang_pilot

cat /workspace/indicservebench_results/sarvam_h100_sglang_pilot/$EXP/human_readable_summary.md
```

## 6. Run 48-Prompt Pilot

Only after smoke succeeds:

```bash
EXP=sarvam_h100_sglang_pilot_$(date -u +%Y%m%dT%H%M%SZ)
PYTHONPATH=src python3 -m indicservebench.h100_sglang_sarvam_pilot \
  --pilot \
  --experiment-id "$EXP" \
  --base-url http://127.0.0.1:30000 \
  --output-root /workspace/indicservebench_results/sarvam_h100_sglang_pilot

cat /workspace/indicservebench_results/sarvam_h100_sglang_pilot/$EXP/human_readable_summary.md
```

## 7. Save Artifacts

```bash
cd /workspace
zip -r sarvam_h100_sglang_pilot_results.zip \
  indicservebench_results/sarvam_h100_sglang_pilot \
  sglang_logs/sarvam_h100_server.log

ls -lh /workspace/sarvam_h100_sglang_pilot_results.zip
```

Download the zip and terminate the pod.

## Meeting Label

Use this exact label:

`preliminary Sarvam-30B-FP8 H100 SGLang streaming pilot, single concurrency, frozen 48-prompt natural suite`

Do not call it production performance.
