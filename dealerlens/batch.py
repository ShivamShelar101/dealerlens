"""Inventory repricing: value a whole lot from a CSV (e.g. a DMS export) and flag mispriced cars."""
import pandas as pd

REQUIRED = ["year", "make", "model", "odometer"]
OPTIONAL = ["stock", "vin", "trim", "condition", "title_status", "state", "drive", "type", "fuel",
            "paint_color", "notes", "list_price", "cost"]


def appraise_inventory(app, df):
    """Returns a copy of df with market values, ranges and a pricing flag per car."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Inventory CSV is missing columns: {', '.join(missing)}")
    df = df.copy()
    cars = []
    for _, r in df.iterrows():
        car = {c: r[c] for c in REQUIRED + OPTIONAL if c in df.columns and pd.notna(r[c]) and c not in ("list_price", "cost")}
        cars.append(car)

    has_both = app.dealer_label and app.private_label
    retail_cars = [{**c, "seller_type": app.dealer_label} if has_both else c for c in cars]
    v, lo, hi = app.predict(retail_cars)
    df["market_retail"], df["retail_low"], df["retail_high"] = v.round(), lo.round(), hi.round()
    if has_both:
        pv, _, _ = app.predict([{**c, "seller_type": app.private_label} for c in cars])
        df["private_party"] = pv.round()
    df["warnings"] = ["; ".join(app.normalize(c)[1]) for c in cars]

    if "list_price" in df:
        lp = pd.to_numeric(df["list_price"], errors="coerce")
        df["vs_market_pct"] = ((lp / df["market_retail"] - 1) * 100).round(1)
        df["pricing_flag"] = "In range"
        df.loc[lp > df["retail_high"], "pricing_flag"] = "Above market: likely slow to sell"
        df.loc[lp < df["retail_low"], "pricing_flag"] = "Below market: leaving money on the table"
        df.loc[lp.isna(), "pricing_flag"] = "No list price"
    if "cost" in df:
        cost = pd.to_numeric(df["cost"], errors="coerce")
        df["expected_gross"] = (df["market_retail"] - cost).round()
    return df


def summarize(df):
    out = {"vehicles": len(df), "total_market_retail": int(df["market_retail"].sum())}
    if "pricing_flag" in df:
        out["flags"] = df["pricing_flag"].value_counts().to_dict()
    if "expected_gross" in df:
        out["total_expected_gross"] = int(df["expected_gross"].sum(skipna=True))
    return out
