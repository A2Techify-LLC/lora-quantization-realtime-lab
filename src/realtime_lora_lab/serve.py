from __future__ import annotations

import argparse
import json
from collections.abc import AsyncIterator

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

from .prompts import SYSTEM_PROMPT, user_prompt
from .schema import IncidentEvent


def create_app(model_path: str) -> FastAPI:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    app = FastAPI(title="Realtime LoRA Incident Router")

    def build_inputs(event: IncidentEvent):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(event)},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, return_tensors="pt").to(model.device)

    @app.post("/route")
    def route(event: IncidentEvent) -> dict:
        inputs = build_inputs(event)
        output = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        return {"raw": decoded.strip()}

    @app.post("/route/stream")
    async def route_stream(event: IncidentEvent) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            inputs = build_inputs(event)
            thread = Thread(
                target=model.generate,
                kwargs={**inputs, "streamer": streamer, "max_new_tokens": 160, "do_sample": False},
            )
            thread.start()
            for token in streamer:
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(args.model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

