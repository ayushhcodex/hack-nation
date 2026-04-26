from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    dataset_path: Path = Path(os.getenv("DATASET_PATH", "data/facilities_india_10k.csv"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "outputs"))
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "")
    mlflow_experiment_name: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "agentic_healthcare_intel"
    )
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    
    use_vector_search: bool = os.getenv("USE_VECTOR_SEARCH", "false").lower() == "true"
    vector_search_endpoint: str = os.getenv("VECTOR_SEARCH_ENDPOINT", "healthintel_endpoint")
    vector_search_index: str = os.getenv("VECTOR_SEARCH_INDEX", "workspace.default.healthintel_deserts")
    
    databricks_host: str = os.getenv("DATABRICKS_HOST", "https://dbc-xxxxxxx.cloud.databricks.com")
    databricks_token: str = os.getenv("DATABRICKS_TOKEN", "")
    llm_endpoint_name: str = os.getenv("LLM_ENDPOINT_NAME", "databricks-meta-llama-3-1-70b-instruct")


settings = Settings()
