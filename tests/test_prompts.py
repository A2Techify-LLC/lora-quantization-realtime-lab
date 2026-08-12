import json

from realtime_lora_lab.prompts import assistant_prompt, user_prompt
from realtime_lora_lab.schema import IncidentEvent, RoutingDecision


def test_user_prompt_contains_event_json():
    event = IncidentEvent(message="API is down", source="prometheus", service="checkout")
    prompt = user_prompt(event)
    assert "Route this event:" in prompt
    assert '"service":"checkout"' in prompt


def test_assistant_prompt_is_compact_valid_json():
    decision = RoutingDecision(
        team="infra",
        priority="p1",
        action="page_on_call",
        summary="Checkout API is down.",
    )
    payload = assistant_prompt(decision)
    assert "\n" not in payload
    assert json.loads(payload)["team"] == "infra"

