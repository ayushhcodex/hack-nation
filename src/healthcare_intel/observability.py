from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore

from healthcare_intel.config import settings


class _NoopSpan:
    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def set_attribute(self, key: str, value: Any) -> None:
        return None


@dataclass
class ObservabilityTracker:
    """MLflow-backed tracking for run-level and step-level observability."""

    run_name: str
    enabled: bool = True

    def setup(self) -> None:
        if not self.enabled or mlflow is None:
            return
        if settings.mlflow_tracking_uri:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

    @contextmanager
    def run(self, **params: Any) -> Iterator[Any]:
        if not self.enabled:
            yield None
            return

        self.setup()
        with mlflow.start_run(run_name=self.run_name):
            if params:
                mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.set_tag("system", "agentic_healthcare_intel")
            mlflow.set_tag("trace.started_utc", datetime.now(timezone.utc).isoformat())
            yield mlflow.active_run()

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Any]:
        if not self.enabled:
            yield _NoopSpan()
            return

        start_span = getattr(mlflow, "start_span", None)
        if callable(start_span):
            with start_span(name=name) as span:
                for k, v in attributes.items():
                    try:
                        span.set_attribute(str(k), v)
                    except Exception:
                        continue
                yield span
            return

        # Fallback for environments without tracing API.
        mlflow.log_param(f"span.{name}.attributes", str(attributes))
        yield _NoopSpan()

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.enabled:
            return
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        if not self.enabled:
            return
        if path.exists():
            mlflow.log_artifact(str(path))


def estimate_trace_cost(num_rows: int, text_chars: int, query_count: int = 0) -> dict[str, float]:
    """
    Lightweight cost proxy for trace analytics.
    This is a conservative heuristic, not billing-grade accounting.
    """
    est_input_tokens = int(text_chars / 4)
    est_query_tokens = int(query_count * 120)
    est_total_tokens = est_input_tokens + est_query_tokens

    # Nominal blended placeholder rate per 1k tokens for planning purposes.
    est_cost_usd = round((est_total_tokens / 1000.0) * 0.002, 6)

    return {
        "rows_processed": float(num_rows),
        "estimated_input_tokens": float(est_input_tokens),
        "estimated_query_tokens": float(est_query_tokens),
        "estimated_total_tokens": float(est_total_tokens),
        "estimated_cost_usd": float(est_cost_usd),
    }
