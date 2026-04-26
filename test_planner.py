import pandas as pd
from healthbricks_india.reasoning.query_engine import run_query
import json
from dotenv import load_dotenv

load_dotenv()

df = pd.read_parquet("outputs/facilities_enriched.parquet")
res = run_query(df, "find me a rural clinic that uses part-time staff and dental services")
print(json.dumps(res.attrs["trace_steps"], indent=2))
