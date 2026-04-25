from __future__ import annotations

import argparse
import json

from healthcare_intel.agents.genie_orchestrator import GenieOrchestrator, GenieTask


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Genie-style autonomous multi-step task")
    parser.add_argument("--action", required=True, help="Task action name")
    parser.add_argument(
        "--params",
        required=True,
        help="JSON object with task parameters",
    )
    args = parser.parse_args()

    task = GenieTask(action=args.action, params=json.loads(args.params))
    orchestrator = GenieOrchestrator()
    result = orchestrator.execute(task)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
