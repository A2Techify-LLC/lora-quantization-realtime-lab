from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Team = Literal["infra", "support", "growth", "data", "sales", "security"]
Priority = Literal["p1", "p2", "p3"]
Action = Literal[
    "page_on_call",
    "create_ticket",
    "rollback_experiment",
    "reply_with_template",
    "contact_customer",
    "disable_account_and_investigate",
    "rerun_after_dependency",
]


class IncidentEvent(BaseModel):
    message: str = Field(min_length=1)
    source: str = "unknown"
    service: str = "unknown"


class RoutingDecision(BaseModel):
    team: Team
    priority: Priority
    action: Action
    summary: str = Field(min_length=1, max_length=240)


class TrainingRecord(IncidentEvent, RoutingDecision):
    pass

