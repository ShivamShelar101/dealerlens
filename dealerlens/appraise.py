"""One call that produces a full appraisal. Used by the API and the UI."""
from dealerlens.deal import deal_score
from dealerlens.depreciation import curve
from dealerlens.nhtsa import decode_vin, risk


def appraise(app, car=None, vin=None, price=None, recon=0.0, with_risk=True):
    car = {k: v for k, v in (car or {}).items() if v not in (None, "")}
    decoded = None
    if vin:
        decoded = decode_vin(vin)
        car = {**{k: v for k, v in decoded.items() if v not in (None, "", "unknown")}, **car}
    missing = [k for k in ("year", "make", "model", "odometer") if not car.get(k)]
    if missing:
        raise ValueError(f"Missing: {', '.join(missing)}")

    _, warnings = app.normalize(car)
    v, lo, hi = app.predict([car])
    out = {
        "car": car, "decoded_from_vin": decoded is not None,
        "value": round(float(v[0])), "low": round(float(lo[0])), "high": round(float(hi[0])),
        "interval": "80% (conformal-calibrated)",
        "factors": app.explain(car),
        "depreciation": curve(app, car),
        "warnings": warnings,
    }
    if price:
        out["deal"] = deal_score(out["value"], out["low"], out["high"], price, recon)
    if with_risk:
        out["risk"] = risk(car)
    return out
