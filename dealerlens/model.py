"""Load the trained models and value cars: point estimate, calibrated range, explanations."""
import json
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from dealerlens import config
from dealerlens.config import CAT_COLS, DATA_YEAR, FEATURES, LABELS
from dealerlens.data import model_key, to_X, trim_key


class Appraiser:
    def __init__(self, folder=config.ARTIFACTS):
        f = pathlib.Path(folder)
        self.meta = json.loads((f / "meta.json").read_text())
        self.point = lgb.Booster(model_file=str(f / "point.txt"))
        self.lo = lgb.Booster(model_file=str(f / "lo.txt"))
        self.hi = lgb.Booster(model_file=str(f / "hi.txt"))
        self.q = self.meta["conformal_q"]
        self.categories = self.meta["categories"]
        self.vocab = self.meta["model_vocab"]  # make -> known model keys
        self.trim_vocab = self.meta.get("trim_vocab", {})  # "make|model" -> known trim keys

    def normalize(self, car):
        """Turn a user/VIN car dict into model features. Returns (features_dict, warnings)."""
        c = {k: v for k, v in car.items() if v not in (None, "")}
        warnings = []
        make = str(c.get("make", "unknown")).lower().strip()
        mk = c.get("model_key") or model_key(c.get("model"))
        if mk not in self.vocab.get(make, []):
            if make not in self.vocab:
                warnings.append(f"Make '{make}' wasn't in the training data; estimate is unreliable.")
            else:
                warnings.append(f"Model '{c.get('model')}' was rare in the training data; using make-level patterns.")
            mk = "other"
        if "age" in c:
            age = float(c["age"])
        else:
            year = int(c["year"])
            if year > DATA_YEAR + 1:
                warnings.append(f"Model year {year} is newer than anything in the training data; estimate is unreliable.")
            age = float(max(DATA_YEAR - year, 0))
        odo = float(c["odometer"])
        # Trim: explicit field (e.g. from the VIN decode) or the word after the model name.
        raw_trim = c.get("trim_key") or (str(c["trim"]).split()[0].lower().replace("-", "") if c.get("trim")
                                         else trim_key(c.get("model"), model_key(c.get("model"))))
        tk = raw_trim if raw_trim in self.trim_vocab.get(f"{make}|{mk}", []) else ("base" if raw_trim == "base" else "other")
        f = {"age": age, "odometer": odo, "miles_per_year": odo / max(age, 0.5),
             "make": make, "model_key": mk, "trim_key": tk}
        for col in CAT_COLS:
            if col not in f:
                f[col] = str(c.get(col, "unknown")).lower().strip()
        return f, warnings

    def _X(self, cars):
        return to_X(pd.DataFrame([self.normalize(c)[0] for c in cars]), self.categories)

    def predict(self, cars):
        """Return arrays (value, low, high) in dollars. [low, high] is a calibrated 80% range."""
        X = self._X(cars)
        p = self.point.predict(X)
        lo = np.minimum(self.lo.predict(X) - self.q, p)
        hi = np.maximum(self.hi.predict(X) + self.q, p)
        return np.exp(p), np.exp(lo), np.exp(hi)

    def explain(self, car, top=6):
        """Top factors with approximate dollar impact (LightGBM's built-in SHAP values)."""
        feats, _ = self.normalize(car)
        X = to_X(pd.DataFrame([feats]), self.categories)
        contrib = self.point.predict(X, pred_contrib=True)[0]
        value = float(np.exp(contrib.sum()))
        out = []
        for name, phi in zip(FEATURES, contrib[:-1]):
            shown = feats[name]
            if name == "odometer":
                shown = f"{shown:,.0f} mi"
            elif name == "miles_per_year":
                shown = f"{shown:,.0f} mi/yr"
            elif name == "age":
                shown = f"{shown:.0f} yr"
            out.append({"feature": LABELS[name], "value": shown,
                        "impact_usd": round(value * (1 - np.exp(-phi)))})  # effect vs. an average car
        return sorted(out, key=lambda r: -abs(r["impact_usd"]))[:top]
