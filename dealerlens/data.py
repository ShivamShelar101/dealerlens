"""Load and clean the Craigslist used-vehicle listings, and split without leakage."""
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from dealerlens.config import CAT_COLS, DATA_YEAR, FEATURES, NUM_COLS
from dealerlens.text import build_text

RAW_COLS = ["id", "price", "year", "manufacturer", "model", "condition", "cylinders", "fuel",
            "odometer", "title_status", "transmission", "drive", "type", "state", "VIN",
            "title", "description", "seller_type", "paint_color", "odometer_broken", "odometer_rolled_over", "url"]
# Deliberately NOT used: monthly_payment (roughly price / 60, so it leaks the answer) and
# image_count / url fields (listing artifacts a dealer appraising a car wouldn't have).


def _flag(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "t"])

# Models whose first word alone is ambiguous ("grand cherokee" vs "grand caravan").
TWO_TOKEN_PREFIXES = {"grand", "town", "model", "land", "range", "santa", "new", "monte", "el"}


def _tokens(name):
    return re.findall(r"[a-z0-9]+", name.lower().replace("-", "")) if isinstance(name, str) else []


def model_key(name):
    """Normalize a free-text model ("Silverado 1500 LT crew cab") to a key ("silverado")."""
    toks = _tokens(name)
    if not toks:
        return "unknown"
    if toks[0] in TWO_TOKEN_PREFIXES and len(toks) > 1:
        return f"{toks[0]} {toks[1]}"
    return toks[0]


def trim_key(name, mkey=None):
    """The word after the model key ("F-150 XLT" -> "xlt", "Silverado 1500 LT" -> "1500"); "base" if none."""
    toks = _tokens(name)
    mkey = mkey or model_key(name)
    rest = toks[len(mkey.split()):] if mkey not in ("unknown", "other") else []
    return rest[0] if rest else "base"


def default_title(year, make, model):
    """What a listing title usually looks like; used when a title is missing and at inference time."""
    y = f"{int(year)} " if year == year and year is not None else ""
    return f"{y}{make or ''} {model or ''}".strip()


def load_raw(path, nrows=None):
    header = pd.read_csv(path, nrows=0).columns
    df = pd.read_csv(path, usecols=[c for c in RAW_COLS if c in header], nrows=nrows, low_memory=False)
    for col in ("price", "year", "odometer"):  # real data has junk values; force numeric
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df


def clean(df, min_model_count=50, min_trim_count=30, max_price_ratio=3.0):
    """Apply cleaning rules. Returns (clean_df, stats) with a count after each rule."""
    d = df.rename(columns={"manufacturer": "make", "VIN": "vin"}).copy()
    for col in ("price", "year", "odometer"):
        d[col] = pd.to_numeric(d[col], errors="coerce").astype(float)
    stats = {"raw": len(d)}

    def step(name, mask):
        nonlocal d
        d = d[mask]
        stats[name] = len(d)

    step("price_1k_150k", d["price"].between(1_000, 150_000))
    step("year_2000_plus", d["year"].between(2000, DATA_YEAR + 1))  # next model year goes on sale early
    step("odometer_500_400k", d["odometer"].between(500, 400_000))
    step("has_make_model", d["make"].notna() & d["model"].notna())
    if "odometer_broken" in d or "odometer_rolled_over" in d:
        bad = pd.Series(False, index=d.index)
        for col in ("odometer_broken", "odometer_rolled_over"):
            if col in d:
                bad |= _flag(d[col])
        step("odometer_not_flagged", ~bad)

    d = d.copy()
    d["make"] = d["make"].str.lower().str.strip()
    d["model_key"] = d["model"].map(model_key)
    d["trim_key"] = [trim_key(m, k) for m, k in zip(d["model"], d["model_key"])]
    d["age"] = (DATA_YEAR - d["year"]).clip(lower=0).astype(float)
    d["miles_per_year"] = d["odometer"] / d["age"].clip(lower=0.5)
    step("plausible_miles_per_year", d["miles_per_year"] < 60_000)
    step("not_new_and_cheap", ~((d["age"] <= 3) & (d["price"] < 3_000)))  # usually lease/down-payment ads

    if "vin" in d:
        has_vin = d["vin"].notna()
        d = pd.concat([d[has_vin].drop_duplicates(["vin", "price", "odometer"]), d[~has_vin]])
    d = d.drop_duplicates(["make", "model", "year", "odometer", "price", "state"])  # reposts
    stats["deduplicated"] = len(d)

    # Fake prices ("$1,500 down", parts cars, typos): far from the median of the same make/model/year.
    grp = d.groupby(["make", "model_key", "year"])["price"]
    ratio = d["price"] / grp.transform("median")
    step("price_near_similar_cars", (grp.transform("size") < 5) | ratio.between(1 / max_price_ratio, max_price_ratio))

    pair = d["make"] + " " + d["model_key"]
    d.loc[pair.map(pair.value_counts()) < min_model_count, "model_key"] = "other"
    trio = d["make"] + " " + d["model_key"] + " " + d["trim_key"]
    d.loc[(trio.map(trio.value_counts()) < min_trim_count) | (d["model_key"] == "other"), "trim_key"] = "other"
    for c in CAT_COLS:
        if c not in d:
            d[c] = "unknown"
        d[c] = d[c].fillna("unknown").astype(str).str.lower().str.strip()
    titles = d["title"] if "title" in d else pd.Series(None, index=d.index, dtype=object)
    descs = d["description"] if "description" in d else pd.Series(None, index=d.index, dtype=object)
    d["text"] = [build_text(t if isinstance(t, str) and t.strip() else default_title(y, mk, m), desc)
                 for t, desc, y, mk, m in zip(titles, descs, d["year"], d["make"], d["model"])]
    d["text_score"] = np.nan  # filled during training by the out-of-fold text model
    d["log_price"] = np.log(d["price"])
    ids = d["id"].astype(str) if "id" in d else pd.Series(d.index.astype(str), index=d.index)
    d["group"] = d["vin"].fillna(ids) if "vin" in d else ids
    return d.reset_index(drop=True), stats


def split(d, seed=42):
    """train / valid (early stopping) / calib (conformal) / test, grouped by VIN so a car never
    appears in two splits."""
    def cut(frame, frac):
        gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
        a, b = next(gss.split(frame, groups=frame["group"]))
        return frame.iloc[a], frame.iloc[b]

    train, rest = cut(d, 0.3)
    train, valid = cut(train, 0.1)
    calib, test = cut(rest, 0.5)
    return train, valid, calib, test


def fit_categories(train):
    return {c: sorted(train[c].astype(str).unique().tolist()) for c in CAT_COLS}


def to_X(d, categories):
    X = d[FEATURES].copy()
    for c in NUM_COLS:
        X[c] = X[c].astype(float)
    for c in CAT_COLS:
        vals = X[c].astype(str)
        vals = vals.where(vals.isin(categories[c]))  # unseen category → missing (LightGBM handles it)
        X[c] = pd.Categorical(vals, categories=categories[c])
    return X
