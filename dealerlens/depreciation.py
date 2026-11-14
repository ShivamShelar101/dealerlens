"""Projected value over time and a 'holds value best/worst' leaderboard."""
import pandas as pd

from dealerlens.config import CAT_COLS, DATA_YEAR


def curve(app, car, years=(0, 1, 2, 3), miles_per_year=None):
    feats, _ = app.normalize(car)
    age, odo = feats["age"], feats["odometer"]
    mpy = miles_per_year or (min(max(odo / age, 5_000), 25_000) if age >= 1 else 12_000)
    rows = [{**feats, "age": age + y, "odometer": odo + y * mpy} for y in years]
    v, lo, hi = app.predict(rows)
    return [{"years_ahead": y, "value": round(float(a)), "low": round(float(b)), "high": round(float(c)),
             "retained_pct": round(float(a / v[0]) * 100, 1)} for y, a, b, c in zip(years, v, lo, hi)]


def leaderboard(app, df, top_n=40, start_age=3, horizon=3, miles_per_year=12_000):
    """For the most common models, % of value retained over `horizon` years for a typical example."""
    d = df[~df["model_key"].isin(["other", "unknown"])]
    pairs = (d["make"] + "|" + d["model_key"]).value_counts().head(top_n).index
    rows = []
    for pair in pairs:
        make, mk = pair.split("|")
        g = d[(d["make"] == make) & (d["model_key"] == mk)]
        typical = {c: g[c].mode().iat[0] for c in CAT_COLS}
        car = {**typical, "make": make, "model_key": mk, "age": start_age,
               "odometer": start_age * miles_per_year}
        pts = curve(app, car, years=(0, horizon), miles_per_year=miles_per_year)
        rows.append({"make": make, "model": mk, "listings": len(g),
                     "value_now": pts[0]["value"], f"value_in_{horizon}y": pts[1]["value"],
                     "retained_pct": pts[1]["retained_pct"]})
    return pd.DataFrame(rows).sort_values("retained_pct", ascending=False).reset_index(drop=True)


__all__ = ["curve", "leaderboard", "DATA_YEAR"]
