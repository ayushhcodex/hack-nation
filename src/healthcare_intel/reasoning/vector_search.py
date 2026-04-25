from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(slots=True)
class VectorSearchConfig:
    endpoint_name: str
    index_name: str
    primary_key: str = "facility_id"
    text_column: str = "full_text"


def _get_client() -> Any:
    try:
        from databricks.vector_search.client import VectorSearchClient
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "databricks-vectorsearch is required for Mosaic AI Vector Search integration"
        ) from exc

    return VectorSearchClient()


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


def sync_index(df: pd.DataFrame, config: VectorSearchConfig) -> None:
    """
    Creates or updates a Mosaic AI Vector Search index from facility rows.
    Requires Databricks runtime/session with proper credentials.
    """
    client = _get_client()
    prepared = _prepare_for_index(df, config.primary_key, config.text_column)

    index = client.get_index(endpoint_name=config.endpoint_name, index_name=config.index_name)

    payload = prepared.to_dict(orient="records")
    index.upsert(payload)


def semantic_search(
    query_text: str,
    config: VectorSearchConfig,
    columns: list[str] | None = None,
    num_results: int = 10,
) -> list[dict[str, Any]]:
    client = _get_client()
    index = client.get_index(endpoint_name=config.endpoint_name, index_name=config.index_name)

    result = index.similarity_search(
        query_text=query_text,
        columns=columns or [config.primary_key, config.text_column],
        num_results=num_results,
    )
    return result.get("result", {}).get("data_array", [])
