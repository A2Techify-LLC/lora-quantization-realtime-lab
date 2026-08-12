from __future__ import annotations

import json

from .schema import IncidentEvent, RoutingDecision, TrainingRecord


SYSTEM_PROMPT = """You are an operations routing model.
Return only valid compact JSON with keys: team, priority, action, summary.
Do not include markdown, explanations, or extra keys."""


def user_prompt(event: IncidentEvent) -> str:
    payload = event.model_dump()
    return "Route this event:\n" + json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def assistant_prompt(decision: RoutingDecision) -> str:
    return json.dumps(decision.model_dump(), separators=(",", ":"), ensure_ascii=False)


def messages_from_record(record: TrainingRecord) -> list[dict[str, str]]:
    event = IncidentEvent(**record.model_dump())
    decision = RoutingDecision(**record.model_dump())
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(event)},
        {"role": "assistant", "content": assistant_prompt(decision)},
    ]


def text_from_record(record: TrainingRecord, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        messages_from_record(record),
        tokenize=False,
        add_generation_prompt=False,
    )

