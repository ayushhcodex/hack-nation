from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from healthcare_intel.config import settings

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchConfig:
    endpoint_name: str = settings.vector_search_endpoint
    index_name: str = settings.vector_search_index
    primary_key: str = "facility_id"
    text_column: str = "full_text"


def _get_client() -> Any:
    try:
        from databricks.vector_search.client import VectorSearchClient
        return VectorSearchClient()
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Vector search client unavailable: {exc}")
        return None


def _prepare_for_index(df: pd.DataFrame, primary_key: str, text_column: str) -> pd.DataFrame:
    work = df.copy()
    if primary_key not in work.columns:
        work[primary_key] = work.index.astype(str)
    work[primary_key] = work[primary_key].astype(str)

    if text_column not in work.columns:
        text_parts = [c for c in ["description", "specialties", "procedure", "equipment", "capability"] if c in work.columns]
        work[text_column] = work[text_parts].fillna("").agg(" ".join, axis=1).str.strip()

    work[text_column] = work[text_column].fillna("").astype(str)
    return work


def sync_index(df: pd.DataFrame, config: VectorSearchConfig | None = None) -> bool:
    """
    Creates or updates a Mosaic AI Vector Search index from facility rows.
    Returns True if sync was theoretically successful.
    """
    config = config or VectorSearchConfig()
    client = _get_client()
    if not client:
        return False
        
    try:
        prepared = _prepare_for_index(df, config.primary_key, config.text_column)
        index = client.get_index(endpoint_name=config.endpoint_name, index_name=config.index_name)
        payload = prepared.to_dict(orient="records")
        index.upsert(payload)
        return True
    except Exception as e:
        logger.error(f"Failed to sync vector index: {e}")
        return False


def semantic_search(
    query_text: str,
    config: VectorSearchConfig | None = None,
    columns: list[str] | None = None,
    num_results: int = 10,
) -> list[dict[str, Any]]:
    config = config or VectorSearchConfig()
    client = _get_client()
    if not client:
        return []
        
    try:
        index = client.get_index(endpoint_name=config.endpoint_name, index_name=config.index_name)
        result = index.similarity_search(
            query_text=query_text,
            columns=columns or [config.primary_key, config.text_column],
            num_results=num_results,
        )

        # Databricks may return either row dicts or row arrays with a separate column manifest.
        data_array = result.get("result", {}).get("data_array", [])
        if not data_array:
            return []

        if isinstance(data_array[0], dict):
            return data_array

        columns_meta = result.get("manifest", {}).get("columns", [])
        col_names = [c.get("name") for c in columns_meta if isinstance(c, dict) and c.get("name")]

        # Fallback to requested output columns when manifest is not present.
        if not col_names:
            col_names = columns or [config.primary_key, config.text_column]

        normalized: list[dict[str, Any]] = []
        for row in data_array:
            if isinstance(row, (list, tuple)):
                normalized.append({k: v for k, v in zip(col_names, row)})
            elif isinstance(row, dict):
                normalized.append(row)
        return normalized
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []
