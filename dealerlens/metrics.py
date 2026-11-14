import numpy as np


def price_metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = np.abs(pred - y)
    return {"mae": float(err.mean()), "median_ae": float(np.median(err)),
            "mape": float((err / y).mean()), "median_ape": float(np.median(err / y)),
            "within_10pct": float((err / y <= 0.10).mean()), "within_20pct": float((err / y <= 0.20).mean())}


def interval_metrics(y, lo, hi):
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return {"coverage": float(((y >= lo) & (y <= hi)).mean()),
            "mean_width": float((hi - lo).mean()),
            "mean_rel_width": float(((hi - lo) / y).mean())}


def conformal_q(lo, hi, y, alpha):
    """Conformalized quantile regression (CQR): how much to widen [lo, hi] so it covers 1-alpha."""
    scores = np.maximum(np.asarray(lo) - y, y - np.asarray(hi))
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def band_index(pred_log, edges):
    """Price band of each prediction (bands are defined on the *predicted* value, known at inference)."""
    return np.searchsorted(np.log(edges), pred_log, side="right")


def mondrian_q(lo, hi, y, pred_log, edges, alpha, min_n=200):
    """Per-band conformal widening. Bands with too few calibration rows fall back to the global value."""
    lo, hi, y, pred_log = (np.asarray(a) for a in (lo, hi, y, pred_log))
    global_q = conformal_q(lo, hi, y, alpha)
    bands = band_index(pred_log, edges)
    qs = []
    for b in range(len(edges) + 1):
        m = bands == b
        qs.append(conformal_q(lo[m], hi[m], y[m], alpha) if m.sum() >= min_n else global_q)
    return qs, global_q
