"""One call that produces a full appraisal. Used by the API and the UI."""
from dealerlens.deal import deal_score
from dealerlens.depreciation import curve
from dealerlens.nhtsa import decode_vin, risk


def _band(app, car):
    v, lo, hi = app.predict([car])
    return {"value": round(float(v[0])), "low": round(float(lo[0])), "high": round(float(hi[0]))}


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
    out = {"car": car, "decoded_from_vin": decoded is not None, "interval": "80% (conformal-calibrated)",
           "warnings": warnings}

    # Two market views when the data has both seller types: what a private seller would ask
    # (roughly the dealer's buy-side) and what a dealer lists it for (retail).
    if app.dealer_label and app.private_label and not car.get("seller_type"):
        retail = _band(app, {**car, "seller_type": app.dealer_label})
        private = _band(app, {**car, "seller_type": app.private_label})
        out.update(retail)
        out["dealer_retail"], out["private_party"] = retail, private
        out["retail_spread"] = retail["value"] - private["value"]
        main = {**car, "seller_type": app.dealer_label}
    else:
        out.update(_band(app, car))
        main = car

    out["factors"] = app.explain(main)
    out["depreciation"] = curve(app, main)
    if car.get("notes"):
        out["text_signals"] = app.text_model.top_phrases(car["notes"])  # only phrases from the notes
    if price:
        out["deal"] = deal_score(out["value"], out["low"], out["high"], price, recon)
    if with_risk:
        out["risk"] = risk(car)
    return out
