# LoRA Quantization Realtime Lab

A practical, end-to-end example of fine-tuning a small instruction model with LoRA, merging the adapter, quantizing the result, and serving it as a real-time JSON incident router.

The example is intentionally concrete: operations events arrive from logs, alerts, or webhooks, and the model returns a strict routing decision:

```json
{
  "team": "infra",
  "priority": "p1",
  "action": "page_on_call",
  "summary": "Public API is returning 5xx errors after deploy."
}
```

This keeps the project useful for production-style learning while staying small enough to run on a single developer machine or low-cost GPU instance.

## What You Learn

- Prepare instruction fine-tuning data as JSONL.
- Fine-tune `Qwen/Qwen2.5-0.5B-Instruct` with LoRA.
- Run low-memory QLoRA training with 4-bit loading when a CUDA GPU is available.
- Merge LoRA adapters back into the base model.
- Quantize the merged model to GGUF with `llama.cpp`.
- Serve a real-time API with FastAPI and Server-Sent Events.
- Measure latency and response quality before/after quantization.

## Project Layout

```text
configs/                    Training and quantization settings
data/                       Tiny example dataset plus generator
docs/blog.md                Extended tutorial/blog
scripts/                    Shell helpers for common workflows
src/realtime_lora_lab/      Python package
tests/                      Lightweight tests for data/prompt behavior
```

## Quick Start

Use Python 3.10, 3.11, or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest
```

Install the ML stack when you are ready to train or serve:

```bash
pip install -e ".[train,serve]"
```

Create a larger synthetic dataset:

```bash
python -m realtime_lora_lab.data.generate_dataset \
  --output data/incidents.train.jsonl \
  --count 1000
```

Run LoRA fine-tuning:

```bash
python -m realtime_lora_lab.train_lora \
  --config configs/lora_qwen_0_5b.yaml
```

Merge the adapter:

```bash
python -m realtime_lora_lab.merge_adapter \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-out outputs/lora-router \
  --merged-out outputs/merged-router
```

Serve the merged model:

```bash
python -m realtime_lora_lab.serve \
  --model outputs/merged-router
```

Send a test event:

```bash
curl -N http://127.0.0.1:8000/route/stream \
  -H 'content-type: application/json' \
  -d '{"message":"Checkout API latency is 8 seconds and error rate jumped to 18% after deploy.","source":"prometheus","service":"checkout"}'
```

## Quantization

This repo uses the widely supported `llama.cpp` path for GGUF quantization.

```bash
git clone https://github.com/ggerganov/llama.cpp external/llama.cpp
cmake -S external/llama.cpp -B external/llama.cpp/build
cmake --build external/llama.cpp/build -j

python external/llama.cpp/convert_hf_to_gguf.py \
  outputs/merged-router \
  --outfile outputs/router-f16.gguf

external/llama.cpp/build/bin/llama-quantize \
  outputs/router-f16.gguf \
  outputs/router-q4_k_m.gguf \
  Q4_K_M
```

The `scripts/quantize_gguf.sh` helper wraps those commands.

## Hardware Notes

The default model is deliberately small. It is meant to demonstrate the full workflow without requiring a 24 GB GPU.

- CPU: possible for data prep, merge, and tiny tests.
- 8-12 GB GPU: comfortable for LoRA/QLoRA on this 0.5B model.
- Larger models: switch `base_model` in the YAML and lower batch size or use gradient accumulation.

## Blog

Read the full walkthrough in [docs/blog.md](docs/blog.md).

## Cost Posture

Everything runs locally after model download. There are no hosted APIs in the default workflow.
