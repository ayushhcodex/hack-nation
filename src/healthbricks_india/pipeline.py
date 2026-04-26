from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from healthbricks_india.agents.multi_pass_extractor import extract_capabilities
from healthbricks_india.analytics.desert_detection import (
    compute_nearest_facility_distances,
    generate_deployment_recommendations,
    identify_specialized_deserts,
)
from healthbricks_india.io.data_loader import load_facility_data
from healthbricks_india.observability import ObservabilityTracker, estimate_trace_cost
from healthbricks_india.trace import TraceCollector
from healthbricks_india.validation.trust_scorer import score_trust, update_trust_with_validation
from healthbricks_india.validation.validator_agent import validate_against_standards
from healthbricks_india.reasoning.vector_search import sync_index
from healthbricks_india.config import settings

import json


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
        # Stage 1: Load data
        trace.add("load_data_started", dataset_path=str(dataset_path))
        with tracker.span("load_data", dataset_path=str(dataset_path)):
            facilities = load_facility_data(dataset_path)
        trace.add("load_data_completed", records=len(facilities), columns=list(facilities.columns[:20]))
        print(f"[Pipeline] Loaded {len(facilities)} facilities with {len(facilities.columns)} columns")

        # Stage 2: Dual-pass capability extraction (keyword + LLM in parallel)
        trace.add("extract_capabilities_started", method="dual_pass_parallel")
        with tracker.span("extract_capabilities"):
            facilities = extract_capabilities(facilities)
        cap_cols = [c for c in facilities.columns if c.startswith("has_") and facilities[c].dtype == bool]
        cap_summary = {c: int(facilities[c].sum()) for c in cap_cols[:15]}
        trace.add("extract_capabilities_completed", capabilities_found=cap_summary)
        print(f"[Pipeline] Extracted capabilities: {len(cap_cols)} flags across {len(facilities)} facilities")

        # Stage 3: Trust scoring with contradiction detection (vectorized)
        trace.add("score_trust_started", method="vectorized_bayesian_8_rules")
        with tracker.span("score_trust"):
            facilities = score_trust(facilities)
        trust_dist = facilities["trust_band"].value_counts().to_dict()
        avg_trust = round(float(facilities["trust_score"].mean()), 4)
        trace.add("score_trust_completed", trust_distribution=trust_dist, avg_trust=avg_trust)
        print(f"[Pipeline] Trust scores: avg={avg_trust}, distribution={trust_dist}")

        # Stage 4: Validator cross-check
        trace.add("validator_started")
        with tracker.span("validator"):
            facilities = validate_against_standards(facilities)
        trace.add("validator_completed")
        print(f"[Pipeline] Validation complete")

        # Stage 4.5: Validator feedback loop — re-adjust trust scores
        trace.add("trust_feedback_loop_started")
        with tracker.span("trust_feedback_loop"):
            pre_feedback_trust = round(float(facilities["trust_score"].mean()), 4)
            facilities = update_trust_with_validation(facilities)
            post_feedback_trust = round(float(facilities["trust_score"].mean()), 4)
        trust_dist_updated = facilities["trust_band"].value_counts().to_dict()
        trace.add(
            "trust_feedback_loop_completed",
            pre_feedback_avg_trust=pre_feedback_trust,
            post_feedback_avg_trust=post_feedback_trust,
            trust_shift=round(post_feedback_trust - pre_feedback_trust, 4),
            updated_distribution=trust_dist_updated,
        )
        print(f"[Pipeline] Trust feedback loop: avg trust {pre_feedback_trust} → {post_feedback_trust}, distribution={trust_dist_updated}")

        # Stage 5: Desert detection
        trace.add("identify_deserts_started")
        with tracker.span("identify_deserts"):
            deserts = identify_specialized_deserts(facilities)
        risk_dist = deserts["risk_tier"].value_counts().to_dict()
        trace.add("identify_deserts_completed", rows=len(deserts), risk_distribution=risk_dist)
        print(f"[Pipeline] Desert regions: {len(deserts)}, risk tiers: {risk_dist}")

        # Stage 6: Nearest facility distances for critical deserts
        trace.add("compute_distances_started")
        with tracker.span("compute_distances"):
            critical_deserts = deserts[deserts["risk_tier"].isin(["critical", "high"])].head(100)
            if not critical_deserts.empty:
                critical_deserts = compute_nearest_facility_distances(critical_deserts, facilities)
        trace.add("compute_distances_completed", critical_count=len(critical_deserts))
        print(f"[Pipeline] Computed distances for {len(critical_deserts)} critical desert regions")

        # Stage 7: Deployment recommendations
        trace.add("generate_recommendations_started")
        recommendations = generate_deployment_recommendations(critical_deserts, top_n=30)
        trace.add("generate_recommendations_completed", count=len(recommendations))
        print(f"[Pipeline] Generated {len(recommendations)} deployment recommendations")

        # Save outputs
        facilities_out = output_dir / "facilities_enriched.parquet"
        deserts_out = output_dir / "specialized_deserts.parquet"
        facilities_csv_out = output_dir / "facilities_enriched.csv"
        deserts_csv_out = output_dir / "specialized_deserts.csv"
        recommendations_out = output_dir / "deployment_recommendations.json"
        critical_deserts_out = output_dir / "critical_deserts.parquet"

        with tracker.span("save_outputs"):
            # Drop parsed list columns that can't be saved to parquet
            save_df = facilities.copy()
            list_cols = [c for c in save_df.columns if c.endswith("_parsed")]
            save_df = save_df.drop(columns=list_cols, errors="ignore")
            
            save_df.to_parquet(facilities_out, index=False)
            save_df.to_csv(facilities_csv_out, index=False)
            deserts.to_parquet(deserts_out, index=False)
            deserts.to_csv(deserts_csv_out, index=False)
            
            if not critical_deserts.empty:
                critical_deserts.to_parquet(critical_deserts_out, index=False)
            
            with open(recommendations_out, "w") as f:
                json.dump(recommendations, f, indent=2, ensure_ascii=False, default=str)

        trace.add(
            "save_outputs_completed",
            facilities_path=str(facilities_out),
            deserts_path=str(deserts_out),
        )

        trace_file = output_dir / "pipeline_trace.jsonl"
        trace.save_jsonl(trace_file)

        tracker.log_metrics({
            "facility_records": float(len(facilities)),
            "desert_regions": float(len(deserts)),
            "critical_deserts": float(len(critical_deserts)),
            "avg_trust_score": avg_trust,
            "recommendations": float(len(recommendations)),
        })

        text_chars = int(facilities["full_text"].fillna("").astype(str).str.len().sum()) if "full_text" in facilities.columns else 0
        tracker.log_metrics(estimate_trace_cost(num_rows=len(facilities), text_chars=text_chars))
        tracker.log_artifact(trace_file)

        # Stage 8: Unity Catalog Volume Sync
        if settings.use_unity_catalog:
            trace.add("sync_unity_catalog_started")
            with tracker.span("sync_unity_catalog"):
                print(f"[Pipeline] Syncing files to Unity Catalog Volume: {settings.uc_volume_path}")
                try:
                    from databricks.sdk import WorkspaceClient
                    import os
                    os.environ["DATABRICKS_HOST"] = settings.databricks_host
                    os.environ["DATABRICKS_TOKEN"] = settings.databricks_token
                    
                    client = WorkspaceClient()
                    
                    with open(facilities_out, "rb") as f:
                        client.files.upload(f"{settings.uc_volume_path}/facilities_enriched.parquet", f, overwrite=True)
                    with open(deserts_out, "rb") as f:
                        client.files.upload(f"{settings.uc_volume_path}/specialized_deserts.parquet", f, overwrite=True)
                    
                    print(f"[Pipeline] Unity Catalog sync successful")
                    trace.add("sync_unity_catalog_completed", success=True)
                except Exception as e:
                    print(f"[Pipeline] Unity Catalog sync failed: {e}")
                    trace.add("sync_unity_catalog_completed", success=False, error=str(e))
        else:
            print("[Pipeline] Unity Catalog sync skipped (USE_UNITY_CATALOG=false)")

        # Stage 9: Vector Search Index Sync
        if settings.use_vector_search:
            trace.add("vector_search_sync_started")
            with tracker.span("vector_search_sync"):
                print(f"[Pipeline] Syncing {len(facilities)} records to Mosaic Vector Search index: {settings.vector_search_index}")
                success = sync_index(facilities)
                if success:
                    print("[Pipeline] Vector index sync initiated successfully")
                    trace.add("vector_search_sync_completed", success=True)
                else:
                    print("[Pipeline] Vector index sync failed. Continuing without it.")
                    trace.add("vector_search_sync_completed", success=False, error="Sync failed")
        else:
            print("[Pipeline] Vector index sync skipped (USE_VECTOR_SEARCH=false)")

    print(f"[Pipeline] Complete! Outputs saved to {output_dir.resolve()}")
    return facilities, deserts

if __name__ == "__main__":
    from healthbricks_india.config import settings
    run_pipeline(
        dataset_path=settings.dataset_path,
        output_dir=settings.output_dir, 
        enable_mlflow=True
    )
