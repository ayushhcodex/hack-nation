from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from healthcare_intel.reasoning.query_engine import run_query

st.set_page_config(page_title="Healthcare Intelligence India", layout="wide")
st.title("Agentic Healthcare Intelligence System")
st.caption("Trust-aware facility discovery and medical desert mapping")

FACILITIES_PATH = Path("outputs/facilities_enriched.parquet")
DESERTS_PATH = Path("outputs/specialized_deserts.parquet")

if not FACILITIES_PATH.exists() or not DESERTS_PATH.exists():
    st.error("Run the pipeline first. Expected outputs in outputs/ directory.")
    st.stop()

facilities = pd.read_parquet(FACILITIES_PATH)
deserts = pd.read_parquet(DESERTS_PATH)

left, right = st.columns([2, 1])
with left:
    query = st.text_input(
        "Natural language query",
        value="Find facility in Bihar for emergency appendectomy with part-time doctors",
    )
with right:
    top_k = st.slider("Top K", min_value=5, max_value=100, value=20, step=5)

if st.button("Run Reasoning Query"):
    results = run_query(facilities, query=query, top_k=top_k)
    st.subheader("Recommended Facilities")
    st.dataframe(results, use_container_width=True)

    if not results.empty:
        st.subheader("Citations (Top Result)")
        cits = results.iloc[0]["citations"]
        if isinstance(cits, str):
            try:
                cits = json.loads(cits)
            except json.JSONDecodeError:
                cits = [cits]
        for idx, citation in enumerate(cits, start=1):
            st.write(f"{idx}. {citation}")

st.subheader("High-Risk Medical Deserts")
st.dataframe(deserts.head(200), use_container_width=True)

if {"latitude", "longitude", "desert_score"}.issubset(facilities.columns):
    map_df = facilities[["latitude", "longitude", "trust_score"]].dropna().copy()
    if not map_df.empty:
        st.subheader("Facility Coverage Map")
        st.map(map_df.rename(columns={"latitude": "lat", "longitude": "lon"}))
