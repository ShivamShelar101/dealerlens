"""Train baselines + LightGBM point/quantile models, calibrate intervals, write reports."""
import json
import pathlib
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dealerlens import config
from dealerlens.config import ALPHA, BAND_EDGES, CAT_COLS, MONOTONE, NUM_COLS
from dealerlens.data import fit_categories, split, to_X
from dealerlens.metrics import band_index, conformal_q, interval_metrics, mondrian_q, price_metrics
from dealerlens.text import TextModel

PARAMS = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, cat_smooth=10, cat_l2=10,
              monotone_constraints=MONOTONE, monotone_constraints_method="advanced", verbose=-1)


def _fit(objective, Xtr, ytr, Xva, yva, rounds, alpha=None, seed=0):
    params = {**PARAMS, "objective": objective, "seed": seed}
    if alpha is not None:
        params["alpha"] = alpha
    if objective == "quantile":  # LightGBM doesn't support monotone constraints for quantile loss
        params.pop("monotone_constraints")
        params.pop("monotone_constraints_method")
    return lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=rounds,
                     valid_sets=[lgb.Dataset(Xva, yva)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def median_baseline(train, test):
    """Median price by make+model+year, falling back to make+year, then make, then global."""
    pred = pd.Series(np.nan, index=test.index)
    for keys in (["make", "model_key", "year"], ["make", "year"], ["make"]):
        med = train.groupby(keys)["price"].median().rename("m")
        fill = test[keys].merge(med, left_on=keys, right_index=True, how="left")["m"]
        pred = pred.fillna(pd.Series(fill.values, index=test.index))
    return pred.fillna(train["price"].median()).values


def ridge_baseline(train, test):
    nums = [c for c in NUM_COLS if c != "text_score"]
    pre = ColumnTransformer([("num", StandardScaler(), nums),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT_COLS)])
    m = make_pipeline(pre, Ridge(alpha=1.0)).fit(train[nums + CAT_COLS], train["log_price"])
    return np.exp(m.predict(test[nums + CAT_COLS]))


def segments(test, value, lo, hi):
    t = test.assign(pred=value, lo=lo, hi=hi)
    t["ape"] = (t["pred"] - t["price"]).abs() / t["price"]
    t["covered"] = (t["price"] >= t["lo"]) & (t["price"] <= t["hi"])
    cuts = {
        "price_band": pd.cut(t["price"], [0, 10_000, 20_000, 35_000, 1e9], labels=["<10k", "10-20k", "20-35k", "35k+"]),
        # What a dealer sees at appraisal time; Mondrian conformal targets coverage within these bands.
        "predicted_band": pd.cut(t["pred"], [0, 10_000, 20_000, 35_000, 1e9], labels=["<10k", "10-20k", "20-35k", "35k+"]),
        "age_band": pd.cut(t["age"], [-1, 3, 7, 12, 100], labels=["0-3y", "4-7y", "8-12y", "13y+"]),
        "mileage_band": pd.cut(t["odometer"], [0, 50_000, 100_000, 150_000, 1e9], labels=["<50k", "50-100k", "100-150k", "150k+"]),
        "type": t["type"].where(t["type"].isin(t["type"].value_counts().head(6).index), "other"),
    }
    rows = []
    for name, groups in cuts.items():
        for g, sub in t.groupby(groups, observed=True):
            rows.append({"segment": name, "group": str(g), "n": len(sub),
                         "mape": round(sub["ape"].mean(), 4), "within_10pct": round((sub["ape"] <= 0.1).mean(), 4),
                         "coverage_80": round(sub["covered"].mean(), 4)})
    return pd.DataFrame(rows)


def train(clean_df, out_dir=config.ARTIFACTS, reports_dir=config.REPORTS, quick=False, log=print):
    t0 = time.time()
    out, rep = pathlib.Path(out_dir), pathlib.Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    rounds = 300 if quick else 5000

    tr, va, ca, te = (d.copy() for d in split(clean_df))
    log(f"split: train={len(tr):,} valid={len(va):,} calib={len(ca):,} test={len(te):,}")

    log("fitting listing-text model (out-of-fold on train)...")
    text_model = TextModel()
    tr["text_score"] = text_model.fit_oof(tr["text"].tolist(), tr["log_price"].values, tr["group"].values)
    for d in (va, ca, te):
        d["text_score"] = text_model.predict(d["text"].tolist())
    text_only_te = price_metrics(te["price"].values, np.exp(te["text_score"].values))
    cats = fit_categories(tr)
    Xtr, Xva, Xca, Xte = (to_X(d, cats) for d in (tr, va, ca, te))
    ytr, yva = tr["log_price"].values, va["log_price"].values

    log("training point model...")
    point = _fit("regression", Xtr, ytr, Xva, yva, rounds)
    log("training quantile models (10th / 90th percentile)...")
    lo_m = _fit("quantile", Xtr, ytr, Xva, yva, rounds, alpha=ALPHA / 2)
    hi_m = _fit("quantile", Xtr, ytr, Xva, yva, rounds, alpha=1 - ALPHA / 2)

    # Conformal calibration on a split the models never saw.
    # Global version (one widening for all cars) and Mondrian version (one per predicted-price band).
    band_qs, q = mondrian_q(lo_m.predict(Xca), hi_m.predict(Xca), ca["log_price"].values,
                            point.predict(Xca), BAND_EDGES, ALPHA)
    p_te = point.predict(Xte)
    raw_lo, raw_hi = lo_m.predict(Xte), hi_m.predict(Xte)
    g_lo, g_hi = np.exp(np.minimum(raw_lo - q, p_te)), np.exp(np.maximum(raw_hi + q, p_te))
    q_row = np.asarray(band_qs)[band_index(p_te, BAND_EDGES)]
    lo = np.exp(np.minimum(raw_lo - q_row, p_te))
    hi = np.exp(np.maximum(raw_hi + q_row, p_te))
    value = np.exp(p_te)

    y = te["price"].values
    results = {
        "baseline_median_make_model_year": price_metrics(y, median_baseline(tr, te)),
        "baseline_ridge": price_metrics(y, ridge_baseline(tr, te)),
        "text_only_ridge": text_only_te,
        "lightgbm": price_metrics(y, value),
        "interval_raw_quantiles": interval_metrics(y, np.exp(raw_lo), np.exp(raw_hi)),
        "interval_conformal_global": interval_metrics(y, g_lo, g_hi),
        "interval_conformal": interval_metrics(y, lo, hi),  # Mondrian (per price band): what the app uses
    }
    meta = {
        "conformal_q": q, "mondrian": {"edges": BAND_EDGES, "q": band_qs}, "categories": cats, "alpha": ALPHA, "data_year": config.DATA_YEAR,
        "model_vocab": {m: sorted(g["model_key"].unique().tolist()) for m, g in tr.groupby("make")},
        "trim_vocab": {f"{m}|{k}": sorted(g["trim_key"].unique().tolist())
                       for (m, k), g in tr.groupby(["make", "model_key"])},
        "rows": {"train": len(tr), "valid": len(va), "calib": len(ca), "test": len(te)},
        "best_iterations": {"point": point.best_iteration, "lo": lo_m.best_iteration, "hi": hi_m.best_iteration},
        "metrics": results, "trained_seconds": round(time.time() - t0),
    }
    text_model.save(out / "text_model.joblib")
    point.save_model(str(out / "point.txt"))
    lo_m.save_model(str(out / "lo.txt"))
    hi_m.save_model(str(out / "hi.txt"))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    te.sample(min(300, len(te)), random_state=1).to_csv(out / "examples.csv", index=False)
    comp_cols = [c for c in ("id", "year", "make", "model_key", "trim_key", "odometer", "price", "state",
                             "seller_type", "title", "url") if c in clean_df]
    clean_df[comp_cols].to_csv(out / "comps.csv.gz", index=False)  # lookup table for comparable listings

    (rep / "metrics.json").write_text(json.dumps(results, indent=1))
    seg = segments(te, value, lo, hi)
    seg.to_csv(rep / "segments.csv", index=False)
    write_metrics_md(results, seg, meta, rep / "metrics.md")

    from dealerlens.depreciation import leaderboard
    from dealerlens.model import Appraiser
    board = leaderboard(Appraiser(out), tr)
    board.to_csv(rep / "leaderboard.csv", index=False)
    log(f"done in {time.time() - t0:.0f}s")
    return results


def write_metrics_md(r, seg, meta, path):
    def row(name, m):
        return (f"| {name} | ${m['mae']:,.0f} | {m['mape']:.1%} | {m['median_ape']:.1%} | "
                f"{m['within_10pct']:.1%} | {m['within_20pct']:.1%} |")
    base = r["baseline_median_make_model_year"]["mae"]
    gain = 1 - r["lightgbm"]["mae"] / base
    lines = [
        "# Results (held-out test set)", "",
        f"Rows: {meta['rows']}", "",
        "| Model | MAE | MAPE | Median APE | Within ±10% | Within ±20% |", "|---|---|---|---|---|---|",
        row("Median by make + model + year", r["baseline_median_make_model_year"]),
        row("Ridge regression (log price)", r["baseline_ridge"]),
        row("Listing text only (TF-IDF ridge)", r["text_only_ridge"]),
        row("**LightGBM + text signal (monotone in age & miles)**", r["lightgbm"]), "",
        f"LightGBM MAE is **{gain:.1%} lower** than the make/model/year median baseline.", "",
        "## 80% price range", "", "| Method | Coverage (target 80%) | Mean width |", "|---|---|---|",
        f"| Raw quantile models | {r['interval_raw_quantiles']['coverage']:.1%} | ${r['interval_raw_quantiles']['mean_width']:,.0f} |",
        f"| Conformal, one global widening | {r['interval_conformal_global']['coverage']:.1%} | ${r['interval_conformal_global']['mean_width']:,.0f} |",
        f"| **Conformal per price band (Mondrian)** | {r['interval_conformal']['coverage']:.1%} | ${r['interval_conformal']['mean_width']:,.0f} |",
        "", "## By segment", "", seg.to_markdown(index=False) if hasattr(seg, "to_markdown") else seg.to_string(index=False),
    ]
    path.write_text("\n".join(lines) + "\n")
