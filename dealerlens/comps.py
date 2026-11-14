"""Comparable listings: real nearby listings for the same model, the first thing a dealer checks."""
import pathlib

import pandas as pd

from dealerlens import config
from dealerlens.data import model_key, trim_key


class Comps:
    def __init__(self, folder=config.ARTIFACTS):
        self.df = pd.read_csv(pathlib.Path(folder) / "comps.csv.gz", low_memory=False)

    def find(self, car, k=8, max_year_gap=2, exclude_id=None):
        make = str(car.get("make", "")).lower().strip()
        mk = car.get("model_key") or model_key(car.get("model"))
        tk = car.get("trim_key") or (str(car["trim"]).split()[0].lower() if car.get("trim")
                                     else trim_key(car.get("model"), mk))
        year, odo = int(car["year"]), float(car["odometer"])
        d = self.df[(self.df["make"] == make) & (self.df["model_key"] == mk)
                    & ((self.df["year"] - year).abs() <= max_year_gap)]
        if exclude_id is not None and "id" in d:
            d = d[d["id"].astype(str) != str(exclude_id)]
        if d.empty:
            return {"count": 0, "items": [], "note": "No comparable listings in the data for this model."}
        state = str(car.get("state", "")).lower()
        score = ((d["year"] - year).abs() + (d["odometer"] - odo).abs() / 20_000
                 + (d["trim_key"] != tk) * 0.5 + (d["state"] != state) * 0.3)
        top = d.assign(_score=score).nsmallest(k, "_score")
        cols = [c for c in ("year", "trim_key", "odometer", "price", "state", "seller_type", "title", "url") if c in top]
        items = top[cols].to_dict("records")
        for it in items:
            it["price"], it["odometer"], it["year"] = int(it["price"]), int(it["odometer"]), int(it["year"])
        return {"count": len(items), "median_price": int(top["price"].median()),
                "price_range": [int(top["price"].min()), int(top["price"].max())], "items": items}
