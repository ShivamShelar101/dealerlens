"""Clean the listings and train everything.

    uv run python -m scripts.train --csv data/vehicles.csv
    uv run python -m scripts.train --csv data/sample.csv --quick     # fast dry run
"""
import argparse
import json

from dealerlens.data import clean, load_raw
from dealerlens.train import train

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--nrows", type=int, help="only read the first N rows (debugging)")
p.add_argument("--quick", action="store_true", help="few boosting rounds")
p.add_argument("--min-model-count", type=int, default=50)
a = p.parse_args()

raw = load_raw(a.csv, a.nrows)
df, stats = clean(raw, a.min_model_count)
print("cleaning (rows remaining after each rule):")
for k, v in stats.items():
    print(f"  {k:28} {v:>9,}")
results = train(df, quick=a.quick)
print(json.dumps(results, indent=1))
print("\nSee reports/metrics.md, reports/segments.csv, reports/leaderboard.csv")
