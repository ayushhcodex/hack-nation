from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TraceCollector:
    """Collects row-level and pipeline-level trace steps."""

    run_id: str
    steps: list[dict[str, Any]] = field(default_factory=list)

    def add(self, step: str, **details: Any) -> None:
        self.steps.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "run_id": self.run_id,
                "step": step,
                "details": details,
            }
        )

    def save_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item in self.steps:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
