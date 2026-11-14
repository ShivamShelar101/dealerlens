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
