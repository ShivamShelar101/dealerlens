"""Fill the README results section from the latest training run (reports/metrics.json + segments.csv).

    uv run python -m scripts.update_readme
"""
import json
import pathlib
import re

import pandas as pd

r = json.loads(pathlib.Path("reports/metrics.json").read_text())
meta = json.loads(pathlib.Path("artifacts/meta.json").read_text())
seg = pd.read_csv("reports/segments.csv")


def row(name, m, bold=False):
    cells = [name, f"${m['mae']:,.0f}", f"{m['mape']:.1%}", f"{m['median_ape']:.1%}", f"{m['within_20pct']:.1%}"]
    if bold:
        cells = [f"**{c}**" for c in cells]
    return "| " + " | ".join(cells) + " |"


gain = 1 - r["lightgbm"]["mae"] / r["baseline_median_make_model_year"]["mae"]
core = seg[(seg["segment"] == "price_band") & seg["group"].isin(["10-20k", "20-35k"])]
core_mape = (core["mape"] * core["n"]).sum() / core["n"].sum()
pb = seg[seg["segment"] == "price_band"].set_index("group")["coverage_80"]
pred_b = seg[seg["segment"] == "predicted_band"].set_index("group")["coverage_80"]

text = f"""## Results (held-out test set: {meta['rows']['test']:,} listings posted Sep-Nov 2026)

| Model | MAE | MAPE | Median APE | Within ±20% |
|---|---|---|---|---|
{row("Median by make + model + year (baseline)", r["baseline_median_make_model_year"])}
{row("Ridge regression (baseline)", r["baseline_ridge"])}
{row("Listing text only (TF-IDF ridge)", r["text_only_ridge"])}
{row("LightGBM + text signal", r["lightgbm"], bold=True)}

- **{gain:.1%} lower MAE** than the make/model/year baseline. Typical dealer inventory ($10k-35k): **{core_mape:.1%} MAPE**.
- **80% price range coverage:** raw quantile models {r['interval_raw_quantiles']['coverage']:.1%} → one global conformal widening {r.get('interval_conformal_global', r['interval_conformal'])['coverage']:.1%} → **per price band (Mondrian) {r['interval_conformal']['coverage']:.1%}**.
- Coverage by **predicted** price band (what a dealer sees at appraisal time): {", ".join(f"{k} {v:.0%}" for k, v in pred_b.items())}.
- Coverage by **true** price band: {", ".join(f"{k} {v:.0%}" for k, v in pb.items())}. Expensive cars tend to be under-predicted (estimates shrink toward the average), so coverage grouped by the true price is lower for $35k+; no method can guarantee coverage conditional on the unknown true price.
- **Iteration history:** v1 26.5% MAPE → v2 20.0% (correct data year, fake-price filter, trim) → v3 {r['lightgbm']['mape']:.1%} (listing text, seller type, odometer flags). v1→v2 also changed the cleaned test set, so part of that gain is data quality.

"""
p = pathlib.Path("README.md")
s = p.read_text()
s = re.sub(r"## Results.*?(?=\n## )", text.rstrip() + "\n", s, count=1, flags=re.S)
p.write_text(s)
print(text)
