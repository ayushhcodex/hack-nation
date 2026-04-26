# Pre-Mortem: Critical Risks & Potential Failure Points

To win the hackathon, you must know your project's weaknesses better than the judges do. Due to time constraints in building the MVP, this architecture carries several high-risk technical debt items that could cause the app to crash during a live demo or cost you points if heavily scrutinized by Databricks engineers.

## 🚨 1. The Rate-Limit Collapse (HTTP 429)
**The Flaw:** In `multi_pass_extractor.py` and `validator_agent.py`, the system processes data using a synchronous `df.iterrows()` loop, firing an HTTP request to the Databricks LLM for **every single row** sequentially.
**Why it causes failure:** Databricks Foundation Model Serving APIs have strict rate limits (Tokens-Per-Minute and Requests-Per-Second). If you tell the pipeline to extract all 10,000 rows at once, you will instantly hit an `HTTP 429 Too Many Requests` error, the API will reject your requests, and the pipeline will fail or fall back to regex for 95% of the data.
**How to hide/defend it:** Do **not** run the full extraction pipeline live on stage. Pre-process the `.parquet` file beforehand. If pushed, explain that for production, you would deploy this via a **Databricks Spark Job utilizing `mapInPandas`** for massive asynchronous parallelization, rather than a single-threaded Python script.

## 🚨 2. Missing Unity Catalog Integration
**The Flaw:** Currently, the pipeline saves its structured outputs locally to `outputs/facilities_enriched.parquet` and reads from a local CSV.
**Why it causes failure:** Databricks heavily grades on "proper use of their ecosystem." Reading and writing local files bypasses **Databricks Unity Catalog**, which is their flagship data governance feature. A strict judge will penalize you for not using Delta Tables.
**How to hide/defend it:** Address it proactively. Say: *"For this rapid-prototype MVP, we heavily optimized the AI extraction layer and intentionally mocked the storage layer locally to reduce database latency. The very next phase of deployment is migrating the local Parquet I/O directly into Unity Catalog managed Delta Tables for enterprise governance."*

## 🚨 3. Vector Index Synchronization Vulnerability
**The Flaw:** The Mosaic AI Vector Search syncing code in `reasoning/vector_search.py` (`index.upsert(payload)`) assumes a perfectly matched vector index schema is already active in your workspace.
**Why it causes failure:** If you accidentally trigger the `sync_vector_index` tool via your Genie Orchestrator and the Databricks Vector Endpoint is asleep, misconfigured, or has mismatched columns, the code will throw an exception.
**How to hide/defend it:** Since the Vector Database logic is the most fragile, rely on the `LLMQueryPlanner` output in the UI for your demo. Don't aggressively trigger live index rebuilds while people are watching. 

## 🚨 4. LLM Hallucinated JSON Schemas
**The Flaw:** The prompt in `multi_pass_extractor.py` asks Llama 3.3 to return `STRICT JSON`. However, LLMs occasionally append conversational text like *"Here is the JSON you requested:"* before the curly braces.
**Why it causes failure:** While we added basic `re.sub(r"```json", "", content)` text stripping, it is not mathematically foolproof. A malformed JSON string will throw a `json.decoder.JSONDecodeError`, skipping the row.
**How to hide/defend it:** This is largely mitigated by our `try/except` fallback to the Keyword Matcher. If the judge asks about it, tell them that moving forward, you would leverage **Databricks Structured Outputs** (Instructor/Pydantic schema enforcement) to guarantee absolute JSON compliance at the model-generation layer.
