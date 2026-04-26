from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

import mlflow
from healthcare_intel.config import settings
from healthcare_intel.pipeline import run_pipeline
from healthcare_intel.reasoning.query_engine import run_query
from healthcare_intel.reasoning.vector_search import VectorSearchConfig, semantic_search, sync_index


@dataclass
class GenieTask:
    """A simple autonomous multi-step task contract for planning workflows."""

    action: str
    params: dict[str, Any]


class GenieOrchestrator:
    """
    Genie-style autonomous orchestrator for multi-step healthcare intelligence tasks.
    """

    @mlflow.trace(name="genie_orchestrator", span_type="AGENT")
    def chat_and_execute(self, prompt: str) -> list[dict[str, Any]]:
        """
        Dynamically plan and execute a sequence of tasks based on a natural language prompt.
        """
        if "api.databricks.com" in settings.databricks_host or "云" in settings.databricks_host or not settings.databricks_token or settings.databricks_host == "https://dbc-xxxxxxx.cloud.databricks.com":
            raise ValueError("Databricks LLM credentials missing. Provide DATABRICKS_HOST to use the autonomous orchestrator.")
            
        url = f"{settings.databricks_host.rstrip('/')}/serving-endpoints/{settings.llm_endpoint_name}/invocations"
        headers = {
            "Authorization": f"Bearer {settings.databricks_token}",
            "Content-Type": "application/json"
        }

        system_instruction = """
You are an autonomous Orchestrator Agent. You control a data pipeline and query engine.
Based on the user's prompt, plan the exact sequence of Tool calls required.

Available Tools (Actions):
1. "build_knowledge_base"
   - Description: Runs the ETL pipeline to process raw hospital notes into structured features and identify deserts.
   - Required Params: "dataset_path" (str)
   - Optional Params: "output_dir" (str)
   
2. "sync_vector_index"
   - Description: Syncs the processed data to Databricks Vector Search.
   - Required Params: "enriched_data_path" (str), "endpoint_name" (str), "index_name" (str)
   
3. "recommend_facilities"
   - Description: Runs a complex structured query and semantic search to find matching facilities.
   - Required Params: "enriched_data_path" (str), "query" (str)
   - Optional Params: "top_k" (int)
   
4. "semantic_retrieve"
   - Description: A fast vector-only retrieval function.
   - Required Params: "query" (str), "endpoint_name" (str), "index_name" (str)

Output Format: Provide a strict JSON array of tasks to execute in order. Do NOT include markdown blocks.
[
  {
    "action": "tool_name",
    "params": {"param1": "value"}
  }
]
"""
        data = {
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1500,
            "temperature": 0.1
        }
        
        try:
            req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=15) as res:
                resp_data = json.loads(res.read().decode("utf-8"))
                content = resp_data["choices"][0]["message"]["content"]
                content = content.replace("```json", "").replace("```", "").strip()
                tasks_data = json.loads(content)
        except Exception as e:
            raise ValueError(f"Orchestrator failed to plan tasks: {e}")
            
        if not isinstance(tasks_data, list):
            raise ValueError(f"Orchestrator returned invalid format. Expected list, got {type(tasks_data)}")
            
        execution_results = []
        for t_dict in tasks_data:
            task_obj = GenieTask(
                action=t_dict.get("action", ""),
                params=t_dict.get("params", {})
            )
            print(f"[Orchestrator] Executing task: {task_obj.action} with params {task_obj.params}")
            try:
                res = self.execute(task_obj)
                execution_results.append({"task": t_dict, "result": res, "status": "success"})
            except Exception as e:
                execution_results.append({"task": t_dict, "error": str(e), "status": "failed"})
                print(f"[Orchestrator] Task failed: {e}")
                break # Stop execution on failure
                
        return execution_results

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
