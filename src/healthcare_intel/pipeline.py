from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from healthcare_intel.agents.extractor_agent import extract_capabilities
from healthcare_intel.analytics.desert_detection import identify_specialized_deserts
from healthcare_intel.io.data_loader import load_facility_csv
from healthcare_intel.observability import ObservabilityTracker, estimate_trace_cost
from healthcare_intel.trace import TraceCollector
from healthcare_intel.validation.trust_scorer import score_trust
from healthcare_intel.validation.validator_agent import validate_against_standards


def run_pipeline(
    dataset_path: Path,
    output_dir: Path,
    enable_mlflow: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    trace = TraceCollector(run_id=run_id)
    tracker = ObservabilityTracker(run_name=run_id, enabled=enable_mlflow)

    with tracker.run(dataset_path=str(dataset_path), run_id=run_id, output_dir=str(output_dir)):
        trace.add("load_data_started", dataset_path=str(dataset_path))
        with tracker.span("load_data", dataset_path=str(dataset_path)):
            facilities = load_facility_csv(dataset_path)
        trace.add("load_data_completed", records=len(facilities), columns=list(facilities.columns))

        trace.add("extract_capabilities_started")
        with tracker.span("extract_capabilities"):
            facilities = extract_capabilities(facilities)
        trace.add("extract_capabilities_completed")

        trace.add("score_trust_started")
        with tracker.span("score_trust"):
            facilities = score_trust(facilities)
        trace.add("score_trust_completed")

        trace.add("validator_started")
        with tracker.span("validator"):
            facilities = validate_against_standards(facilities)
        trace.add("validator_completed")

        trace.add("identify_deserts_started")
        with tracker.span("identify_deserts"):
            deserts = identify_specialized_deserts(facilities)
        trace.add("identify_deserts_completed", rows=len(deserts))

        facilities_out = output_dir / "facilities_enriched.parquet"
        deserts_out = output_dir / "specialized_deserts.parquet"
        facilities_csv_out = output_dir / "facilities_enriched.csv"
        deserts_csv_out = output_dir / "specialized_deserts.csv"

        with tracker.span("save_outputs"):
            facilities.to_parquet(facilities_out, index=False)
            deserts.to_parquet(deserts_out, index=False)
            facilities.to_csv(facilities_csv_out, index=False)
            deserts.to_csv(deserts_csv_out, index=False)

        trace.add(
            "save_outputs_completed",
            facilities_path=str(facilities_out),
            deserts_path=str(deserts_out),
        )

        trace_file = output_dir / "pipeline_trace.jsonl"
        trace.save_jsonl(trace_file)

        tracker.log_metrics(
            {
                "facility_records": float(len(facilities)),
                "desert_regions": float(len(deserts)),
            }
        )

        text_chars = int(facilities["full_text"].fillna("").astype(str).str.len().sum()) if "full_text" in facilities.columns else 0
        tracker.log_metrics(estimate_trace_cost(num_rows=len(facilities), text_chars=text_chars))
        tracker.log_artifact(trace_file)

    return facilities, deserts
