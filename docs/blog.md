# LoRA Fine-Tuning and Quantization: A Real-Time Incident Router From First Principles

Large language models are impressive out of the box, but production systems rarely need a model to know everything. They need a model to do one job reliably, cheaply, and fast. This project shows that workflow end to end: take a small instruction model, fine-tune it with LoRA for a concrete operations task, merge the adapter, quantize the model, and serve it as a real-time routing API.

The example is an incident router. It reads incoming operational events from sources like Prometheus, Datadog, PostHog, Zendesk, Auth0, or email, then returns strict JSON:

```json
{
  "team": "infra",
  "priority": "p1",
  "action": "page_on_call",
  "summary": "Checkout API is slow and returning elevated 5xx errors after deploy."
}
```

This is a better demonstration than a generic sentiment classifier because it has the shape of a real internal automation. It needs structured output, domain-specific labels, low latency, and predictable behavior. Those are exactly the reasons teams reach for LoRA and quantization.

## The Problem

Imagine a small SaaS team. Alerts and messages arrive constantly:

- Metrics alerts report latency, error rate, or database pressure.
- Customer support tickets report broken workflows.
- Product analytics show conversion drops after experiments.
- Security tools report suspicious logins.
- Sales emails ask for vendor review material.

Routing those events manually is boring and slow. Routing them with a giant hosted model is easy, but it introduces recurring API cost, external dependency, and data-sharing concerns. A specialized local model can sit in the middle: small enough to host, trained enough to follow house rules, and cheap enough to run all day.

## Why LoRA

Full fine-tuning updates every weight in a model. That is expensive because optimizer states and gradients multiply memory use. LoRA, short for Low-Rank Adaptation, freezes the base model and trains small adapter matrices inside selected layers. Instead of rewriting the whole model, you teach it a focused behavior through a compact set of additional weights.

For this project, LoRA is a good fit because the target behavior is narrow:

- Read an incident event.
- Choose one team from a small set.
- Choose one priority.
- Choose one action.
- Write a concise summary.
- Return only valid JSON.

We do not need to teach the model a new language or huge new world knowledge. We need it to obey a schema and internal routing policy. That is adapter territory.

## Why Quantization

After fine-tuning, the model still needs to run. Quantization compresses model weights to lower precision so inference uses less memory and often runs faster. Instead of keeping every weight in 16-bit or 32-bit precision, we can store it in a compact format such as 4-bit GGUF.

Quantization is the second half of the cost story:

- LoRA makes training affordable.
- Quantization makes serving affordable.

The workflow in this repo uses Hugging Face and PEFT for LoRA, then `llama.cpp` for GGUF quantization. That path is popular because GGUF models are easy to move between machines and run locally.

## The Base Model

The default config uses:

```yaml
base_model: Qwen/Qwen2.5-0.5B-Instruct
```

This is not the only model you can use. It is chosen because it is small enough for a tutorial and realistic enough to demonstrate instruction formatting, adapter training, merging, and quantization. On stronger hardware, you can change the base model to a 1.5B, 3B, 7B, or larger instruction model and keep the rest of the project structure.

## The Data Format

Training data lives in JSONL. Each line has the input event and the expected routing output:

```json
{"message":"Database CPU is 96% for 12 minutes and write latency is above the SLO.","source":"datadog","service":"orders-db","team":"data","priority":"p1","action":"page_on_call","summary":"Orders database CPU and write latency are breaching SLO."}
```

During training, each record is converted into chat messages:

```text
system: You are an operations routing model. Return only valid compact JSON...
user: Route this event: {"message":"...","source":"datadog","service":"orders-db"}
assistant: {"team":"data","priority":"p1","action":"page_on_call","summary":"..."}
```

This mirrors how the model will be called in production. That matters. Fine-tuning examples should look like inference examples, otherwise you teach one behavior and deploy another.

## Training

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train,serve,test]"
```

Generate more training rows:

```bash
python -m realtime_lora_lab.data.generate_dataset \
  --output data/incidents.train.jsonl \
  --count 1000
```

Then update `configs/lora_qwen_0_5b.yaml` to point at `data/incidents.train.jsonl`, or use the included sample file for a smoke test.

Run training:

```bash
python -m realtime_lora_lab.train_lora \
  --config configs/lora_qwen_0_5b.yaml
```

The important LoRA settings are:

```yaml
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

`r` controls adapter rank. Higher rank gives the adapter more capacity but costs more memory. `alpha` scales the adapter update. The target modules choose which transformer projections receive adapters. For instruction behavior and JSON routing, attention and MLP projections are a sensible default.

The config also enables 4-bit model loading when CUDA is available:

```yaml
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
```

That is QLoRA-style training. The base model is loaded in 4-bit form, while LoRA adapter weights are trained. This dramatically lowers GPU memory needs.

## Merging the Adapter

LoRA training produces an adapter directory, not a complete standalone model. For many serving paths, you can load the base model plus adapter. For quantization, it is usually cleaner to merge the adapter into the base model first:

```bash
python -m realtime_lora_lab.merge_adapter \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-out outputs/lora-router \
  --merged-out outputs/merged-router
```

The merged model contains the base weights plus the LoRA delta. This is the checkpoint you export to GGUF.

## Quantizing to GGUF

Clone and build `llama.cpp`:

```bash
git clone https://github.com/ggerganov/llama.cpp external/llama.cpp
cmake -S external/llama.cpp -B external/llama.cpp/build
cmake --build external/llama.cpp/build -j
```

Convert the merged Hugging Face model:

```bash
python external/llama.cpp/convert_hf_to_gguf.py \
  outputs/merged-router \
  --outfile outputs/router-f16.gguf
```

Quantize it:

```bash
external/llama.cpp/build/bin/llama-quantize \
  outputs/router-f16.gguf \
  outputs/router-q4_k_m.gguf \
  Q4_K_M
```

The helper script wraps this:

```bash
scripts/quantize_gguf.sh outputs/merged-router outputs
```

`Q4_K_M` is a practical default. It gives a strong compression/quality balance for many local deployments. You should still evaluate it against your own data because quantization can affect structured-output reliability.

## Serving in Real Time

The FastAPI server exposes two endpoints:

- `POST /route` returns a completed response.
- `POST /route/stream` streams generated tokens with Server-Sent Events.

Start the server:

```bash
python -m realtime_lora_lab.serve \
  --model outputs/merged-router
```

Stream a request:

```bash
curl -N http://127.0.0.1:8000/route/stream \
  -H 'content-type: application/json' \
  -d '{"message":"Checkout API latency is 8 seconds and error rate jumped to 18% after deploy.","source":"prometheus","service":"checkout"}'
```

Streaming matters for real-time interfaces because users get immediate feedback. Even if the final JSON takes a moment, the system feels alive.

## Evaluation

Do not trust a fine-tuned model just because the loss went down. For this project, useful evaluation should include:

1. JSON validity rate.
2. Team classification accuracy.
3. Priority accuracy.
4. Action accuracy.
5. Summary length and usefulness.
6. Latency before and after quantization.
7. Failure behavior on ambiguous or out-of-distribution events.

The easiest first metric is JSON validity. If the model cannot reliably produce parseable JSON, fix that before debating model quality. Add more examples with strict formatting. Keep the system prompt short. Avoid output prose. Consider constrained decoding in production.

## Common Failure Modes

The model may invent new teams or actions. That usually means the schema was not repeated enough in the examples, or the base model is too loosely prompted.

The model may return markdown. That means your assistant examples are too permissive or your inference prompt differs from training.

The model may over-page `p1`. That means the dataset has priority imbalance. Add more `p2` and `p3` examples with clear contrast.

The quantized model may be slightly worse at strict JSON. Add a post-generation validator and retry once with a repair prompt, or use constrained generation for the final deployment.

## Production Hardening

For a real deployment, add:

- Authentication around the routing endpoint.
- Request logging with sensitive-field filtering.
- A schema validator before any downstream action.
- Human confirmation for irreversible actions.
- A fallback queue when the model output is invalid.
- Offline evaluation before each adapter or quantized model promotion.
- Versioned model artifacts.

The model should recommend actions. It should not page, disable accounts, or email customers without system-level guardrails.

## Why This Reduces Cost

Hosted frontier models are excellent, but many internal workflows do not need frontier reasoning on every request. The cost reducer is specialization:

- A small base model.
- A focused LoRA adapter.
- Local inference.
- Quantized weights.
- Strict schema output.

The result is a model that can run continuously for a narrow task without paying per-token API fees. You can still escalate hard cases to a larger model. That hybrid pattern is usually the best architecture: local model for the common path, larger model for exceptions.

## Where to Go Next

Once the full workflow runs, improve it in this order:

1. Replace synthetic data with real historical incidents after removing secrets.
2. Add a proper validation split.
3. Track JSON validity and routing accuracy.
4. Try a larger base model.
5. Compare LoRA ranks.
6. Compare GGUF quantization levels.
7. Add a small UI that streams routing decisions live.

LoRA and quantization are not magic tricks. They are engineering tools. Used together, they let you turn a general model into a cheap, local, focused worker that does one job well.

