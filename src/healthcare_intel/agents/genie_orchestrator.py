from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from healthcare_intel.pipeline import run_pipeline
from healthcare_intel.reasoning.query_engine import run_query
from healthcare_intel.reasoning.vector_search import VectorSearchConfig, semantic_search, sync_index


@dataclass(slots=True)
class GenieTask:
    """A simple autonomous multi-step task contract for planning workflows."""

    action: str
    params: dict[str, Any]


class GenieOrchestrator:
    """
    Genie-style autonomous orchestrator for multi-step healthcare intelligence tasks.
    """

    def execute(self, task: GenieTask) -> dict[str, Any]:
        action = task.action.strip().lower()

        if action == "build_knowledge_base":
            dataset = Path(task.params["dataset_path"])
            output = Path(task.params.get("output_dir", "outputs"))
            facilities, deserts = run_pipeline(
                dataset,
                output,
                enable_mlflow=bool(task.params.get("enable_mlflow", True)),
            )
            return {
                "status": "ok",
                "action": action,
                "facilities": len(facilities),
                "deserts": len(deserts),
                "output_dir": str(output),
            }

        if action == "recommend_facilities":
            frame = pd.read_parquet(task.params["enriched_data_path"])
            results = run_query(
                frame,
                query=task.params["query"],
                latitude=task.params.get("latitude"),
                longitude=task.params.get("longitude"),
                top_k=int(task.params.get("top_k", 20)),
            )
            return {
                "status": "ok",
                "action": action,
                "count": len(results),
                "results": results.to_dict(orient="records"),
            }

        if action == "sync_vector_index":
            frame = pd.read_parquet(task.params["enriched_data_path"])
            cfg = VectorSearchConfig(
                endpoint_name=task.params["endpoint_name"],
                index_name=task.params["index_name"],
                primary_key=task.params.get("primary_key", "facility_id"),
                text_column=task.params.get("text_column", "full_text"),
            )
            sync_index(frame, cfg)
            return {"status": "ok", "action": action}

        if action == "semantic_retrieve":
            cfg = VectorSearchConfig(
                endpoint_name=task.params["endpoint_name"],
                index_name=task.params["index_name"],
                primary_key=task.params.get("primary_key", "facility_id"),
                text_column=task.params.get("text_column", "full_text"),
            )
            hits = semantic_search(
                task.params["query"],
                config=cfg,
                columns=task.params.get("columns"),
                num_results=int(task.params.get("num_results", 10)),
            )
            return {"status": "ok", "action": action, "hits": hits}

        raise ValueError(f"Unsupported Genie action: {task.action}")
