from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


TEMPLATES = [
    ("infra", "p1", "page_on_call", "{service} has {metric} above {value} after deploy.", "{service} is breaching {metric} after deploy."),
    ("support", "p3", "create_ticket", "Customer cannot {task} in {service}.", "Customer needs help with {task} in {service}."),
    ("growth", "p2", "rollback_experiment", "{service} experiment caused {metric} to drop {value}.", "{service} experiment correlates with {metric} drop."),
    ("data", "p1", "page_on_call", "{service} database reports {metric} above {value}.", "{service} database is breaching {metric}."),
    ("security", "p1", "disable_account_and_investigate", "Suspicious admin login from {source} on {service}.", "Suspicious admin login needs immediate investigation."),
    ("sales", "p3", "reply_with_template", "Prospect asks for {task} during vendor review.", "Prospect requested {task}."),
]

SERVICES = ["checkout", "billing-portal", "orders-db", "identity", "website", "analytics", "integrations"]
METRICS = ["latency", "error rate", "CPU", "signup conversion", "write latency"]
VALUES = ["8 seconds", "18%", "96%", "40%", "the SLO"]
TASKS = ["update invoice address", "SOC 2 documentation", "reset SSO", "export reports", "change account owner"]
SOURCES = ["prometheus", "datadog", "zendesk", "posthog", "email", "auth0", "airflow"]


def build_record() -> dict[str, str]:
    team, priority, action, message_template, summary_template = random.choice(TEMPLATES)
    values = {
        "service": random.choice(SERVICES),
        "metric": random.choice(METRICS),
        "value": random.choice(VALUES),
        "task": random.choice(TASKS),
        "source": random.choice(SOURCES),
    }
    return {
        "message": message_template.format(**values),
        "source": values["source"],
        "service": values["service"],
        "team": team,
        "priority": priority,
        "action": action,
        "summary": summary_template.format(**values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for _ in range(args.count):
            handle.write(json.dumps(build_record(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

