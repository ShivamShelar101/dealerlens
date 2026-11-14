"""Friday-night EDA: plots + a text summary that justify the cleaning rules.

    uv run python -m scripts.eda --csv data/vehicles.csv
Writes reports/eda/*.png and reports/eda/summary.md
"""
import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dealerlens.data import clean, load_raw  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--nrows", type=int)
a = p.parse_args()

out = pathlib.Path("reports/eda")
out.mkdir(parents=True, exist_ok=True)
raw = load_raw(a.csv, a.nrows)
d, stats = clean(raw)


def save(name):
    plt.tight_layout()
    plt.savefig(out / f"{name}.png", dpi=120)
    plt.close()


# 1. Raw price distribution (shows the junk: $0, $1, $999,999,999 listings)
plt.figure(figsize=(7, 4))
plt.hist(np.log10(raw["price"].clip(lower=1)), bins=100)
plt.xlabel("log10(listed price)"); plt.ylabel("listings"); plt.title("Raw prices (before cleaning)")
save("1_raw_price")

# 2. Clean price distribution
plt.figure(figsize=(7, 4))
plt.hist(d["price"], bins=100)
plt.xlabel("price ($)"); plt.title("Prices after cleaning")
save("2_clean_price")

# 3. Depreciation: median price by age
plt.figure(figsize=(7, 4))
d.groupby("age")["price"].median().plot(marker="o")
plt.xlabel("age (years)"); plt.ylabel("median price ($)"); plt.title("Median price by vehicle age")
save("3_price_by_age")

# 4. Mileage effect
plt.figure(figsize=(7, 4))
bins = np.arange(0, 300_001, 20_000)
d.groupby(np.digitize(d["odometer"], bins))["price"].median().plot(marker="o")
plt.xlabel("mileage bucket (20k miles each)"); plt.ylabel("median price ($)"); plt.title("Median price by mileage")
save("4_price_by_mileage")

# 5. Missingness per column
plt.figure(figsize=(7, 4))
(raw.isna().mean() * 100).sort_values().plot(kind="barh")
plt.xlabel("% missing"); plt.title("Missing values in raw data")
save("5_missing")

top = (d["make"] + " " + d["model_key"]).value_counts().head(15)
lines = ["# EDA summary", "", "## Cleaning funnel", "", "| Rule | Rows left |", "|---|---|"]
lines += [f"| {k} | {v:,} |" for k, v in stats.items()]
lines += ["", f"Kept {stats[list(stats)[-1]] / stats['raw']:.1%} of raw rows.", "",
          "## Top 15 make/model", "", "| Make / model | Listings |", "|---|---|"]
lines += [f"| {k} | {v:,} |" for k, v in top.items()]
lines += ["", "## Clean price", "", d["price"].describe().round(0).to_string()]
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nPlots in {out}/")
