from __future__ import annotations

import argparse
from pathlib import Path

from healthcare_intel.pipeline import run_pipeline
from healthcare_intel.reasoning.vector_search import VectorSearchConfig, sync_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agentic Healthcare Intelligence pipeline")
    parser.add_argument("--dataset", required=True, help="Path to input CSV")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow run and tracing instrumentation",
    )
    parser.add_argument(
        "--vector-endpoint",
        default="",
        help="Databricks Mosaic Vector Search endpoint name",
    )
    parser.add_argument(
        "--vector-index",
        default="",
        help="Databricks Mosaic Vector Search index name",
    )
    args = parser.parse_args()

    facilities, deserts = run_pipeline(
        Path(args.dataset),
        Path(args.output),
        enable_mlflow=not args.disable_mlflow,
    )

    if args.vector_endpoint and args.vector_index:
        cfg = VectorSearchConfig(
            endpoint_name=args.vector_endpoint,
            index_name=args.vector_index,
            primary_key="facility_id",
            text_column="full_text",
        )
        sync_index(facilities, cfg)
        print("Vector index sync: completed")

    print(f"Enriched facilities: {len(facilities)}")
    print(f"Desert regions: {len(deserts)}")
    print(f"Saved outputs to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
