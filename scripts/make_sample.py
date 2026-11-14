"""Generate synthetic listings in the same format as the Craigslist CSV.

Use it to test the whole pipeline before the real data finishes downloading. Not for results.

    uv run python -m scripts.make_sample --n 20000 --out data/sample.csv
"""
import argparse

import numpy as np
import pandas as pd

from dealerlens.config import DATA_YEAR

CARS = [  # make, model, base price new, yearly depreciation, type, cylinders, drive
    ("toyota", "camry", 27000, 0.10, "sedan", "4 cylinders", "fwd"),
    ("toyota", "tacoma", 33000, 0.07, "pickup", "6 cylinders", "4wd"),
    ("honda", "civic", 24000, 0.10, "sedan", "4 cylinders", "fwd"),
    ("honda", "cr-v", 29000, 0.10, "SUV", "4 cylinders", "4wd"),
    ("ford", "f-150", 40000, 0.11, "pickup", "8 cylinders", "4wd"),
    ("ford", "escape", 27000, 0.14, "SUV", "4 cylinders", "fwd"),
    ("chevrolet", "silverado 1500", 41000, 0.11, "pickup", "8 cylinders", "4wd"),
    ("chevrolet", "malibu", 24000, 0.15, "sedan", "4 cylinders", "fwd"),
    ("bmw", "3 series", 43000, 0.16, "sedan", "4 cylinders", "rwd"),
    ("jeep", "grand cherokee", 38000, 0.13, "SUV", "6 cylinders", "4wd"),
    ("nissan", "altima", 25000, 0.15, "sedan", "4 cylinders", "fwd"),
    ("tesla", "model 3", 45000, 0.12, "sedan", "other", "rwd"),
]
COND = {"excellent": 1.05, "good": 1.0, "like new": 1.1, "fair": 0.8, "salvage": 0.5}
STATES = ["ca", "tx", "fl", "ny", "wa", "oh", "co", "az"]

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=20000)
p.add_argument("--out", default="data/sample.csv")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

rng = np.random.default_rng(a.seed)
rows = []
for i in range(a.n):
    make, model, base, dep, typ, cyl, drive = CARS[rng.integers(len(CARS))]
    year = int(rng.integers(DATA_YEAR - 18, DATA_YEAR + 1))
    age = DATA_YEAR - year
    odo = max(500, rng.normal(12_000, 4_000) * max(age, 0.3))
    cond = rng.choice(list(COND), p=[0.3, 0.4, 0.15, 0.1, 0.05])
    title = rng.choice(["clean", "rebuilt", "salvage"], p=[0.93, 0.05, 0.02])
    price = base * (1 - dep) ** age * np.exp(-odo / 250_000) * COND[cond]
    price *= {"clean": 1.0, "rebuilt": 0.75, "salvage": 0.55}[title]
    price *= np.exp(rng.normal(0, 0.15))
    rows.append({"id": i, "price": round(price), "year": year, "manufacturer": make, "model": model,
                 "condition": cond, "cylinders": cyl, "fuel": "electric" if make == "tesla" else "gas",
                 "odometer": round(odo), "title_status": title, "transmission": "automatic",
                 "drive": drive, "type": typ, "state": rng.choice(STATES), "VIN": f"SYN{i:014d}"})
pd.DataFrame(rows).to_csv(a.out, index=False)
print(f"wrote {a.n:,} synthetic listings to {a.out}")
