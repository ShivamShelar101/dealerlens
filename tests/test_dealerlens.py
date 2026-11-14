import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from dealerlens import nhtsa
from dealerlens.data import clean, model_key, split, trim_key
from dealerlens.deal import deal_score
from dealerlens.metrics import conformal_q


# ---------- data ----------
@pytest.mark.parametrize("raw,key", [
    ("Silverado 1500 LT", "silverado"), ("F-150 XLT", "f150"), ("grand cherokee laredo", "grand cherokee"),
    ("CR-V EX", "crv"), (None, "unknown"), ("", "unknown")])
def test_model_key(raw, key):
    assert model_key(raw) == key


@pytest.mark.parametrize("raw,trim", [
    ("F-150 XLT", "xlt"), ("Silverado 1500 LT", "1500"), ("grand cherokee laredo", "laredo"),
    ("camry", "base"), (None, "base")])
def test_trim_key(raw, trim):
    assert trim_key(raw) == trim


def raw_df():
    base = {"price": 15000, "year": 2015, "manufacturer": "Toyota", "model": "Camry SE", "condition": "good",
            "cylinders": "4 cylinders", "fuel": "gas", "odometer": 80000, "title_status": "clean",
            "transmission": "automatic", "drive": "fwd", "type": "sedan", "state": "ca"}
    rows = [{**base, "id": i, "VIN": f"V{i}", "odometer": 80000 + i} for i in range(60)]
    rows += [
        {**base, "id": 100, "VIN": "BAD1", "price": 500},             # too cheap
        {**base, "id": 101, "VIN": "BAD2", "year": 1995},             # too old
        {**base, "id": 102, "VIN": "BAD3", "odometer": 900_000},      # impossible mileage
        {**base, "id": 103, "VIN": "BAD4", "year": 2025, "price": 2000},  # new and cheap → lease ad
        {**base, "id": 104, "VIN": "V0", "odometer": 80000},           # duplicate of id 0
        {**base, "id": 105, "VIN": "BAD5", "price": 1500, "odometer": 81234},  # "$1,500 down" ad: 10x below similar cars
    ]
    return pd.DataFrame(rows)


def test_clean_rules():
    d, stats = clean(raw_df(), min_model_count=10)
    assert len(d) == 60
    assert d["make"].eq("toyota").all() and d["model_key"].eq("camry").all()
    assert stats["raw"] == 66
    assert stats["deduplicated"] - stats["price_near_similar_cars"] == 1  # the fake price


def test_rare_models_become_other():
    d, _ = clean(raw_df(), min_model_count=1000)
    assert d["model_key"].eq("other").all()


def test_split_has_no_vin_leakage():
    d, _ = clean(raw_df(), min_model_count=10)
    d = pd.concat([d, d.assign(id=d["id"] + 1000, price=d["price"] + 1)])  # same VINs twice
    parts = split(d)
    groups = [set(p["group"]) for p in parts]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not groups[i] & groups[j]


# ---------- conformal ----------
def test_conformal_reaches_target_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 20000)
    lo, hi = np.full_like(y, -0.5), np.full_like(y, 0.5)  # deliberately too narrow
    q = conformal_q(lo[:10000], hi[:10000], y[:10000], alpha=0.2)
    cov = ((y[10000:] >= lo[10000:] - q) & (y[10000:] <= hi[10000:] + q)).mean()
    assert 0.78 <= cov <= 0.82


# ---------- deal ----------
def test_deal_labels():
    assert deal_score(20000, 17000, 23000, price=15000)["label"] == "Buy"
    assert deal_score(20000, 17000, 23000, price=22000)["label"] == "Pass"
    assert deal_score(20000, 17000, 23000, price=19000)["label"] == "Negotiate"
    assert deal_score(20000, 17000, 23000, price=15000, recon=6000)["label"] == "Pass"


# ---------- NHTSA ----------
VPIC_SAMPLE = {"ModelYear": "2018", "Make": "HONDA", "Model": "Civic", "Trim": "EX",
               "EngineCylinders": "4", "FuelTypePrimary": "Gasoline", "DriveType": "FWD/Front-Wheel Drive",
               "BodyClass": "Sedan/Saloon", "TransmissionStyle": "Continuously Variable Transmission (CVT)"}


def test_vpic_mapping():
    car = nhtsa.vpic_to_car(VPIC_SAMPLE)
    assert car == {"year": 2018, "make": "honda", "model": "Civic", "trim": "EX", "cylinders": "4 cylinders",
                   "fuel": "gas", "drive": "fwd", "type": "sedan", "transmission": "automatic"}


def test_invalid_vin_rejected():
    with pytest.raises(ValueError):
        nhtsa.decode_vin("1HGCM82633A00435O")  # contains O


def test_recalls_and_risk(monkeypatch):
    def fake_get(url, params=None):
        if "recalls" in url:
            return {"Count": 2, "results": [{"NHTSACampaignNumber": "18V001000", "Component": "AIR BAGS", "Summary": "x"}] * 2}
        return {"count": 150, "results": [{"components": "ENGINE, ELECTRICAL SYSTEM"}] * 150}
    monkeypatch.setattr(nhtsa, "_get", fake_get)
    nhtsa.recalls.cache_clear(); nhtsa.complaints.cache_clear()
    r = nhtsa.risk({"make": "honda", "model": "Civic", "year": 2018})
    assert r["recalls"]["count"] == 2 and r["complaints"]["count"] == 150
    assert len(r["flags"]) == 2


def test_nhtsa_failure_is_graceful(monkeypatch):
    def boom(url, params=None):
        raise ConnectionError("offline")
    monkeypatch.setattr(nhtsa, "_get", boom)
    nhtsa.recalls.cache_clear(); nhtsa.complaints.cache_clear()
    r = nhtsa.risk({"make": "ford", "model": "Escape", "year": 2017})
    assert r["flags"] == [] and r["recalls"]["count"] is None


# ---------- end to end on synthetic data ----------
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    csv = tmp / "sample.csv"
    subprocess.run([sys.executable, "-m", "scripts.make_sample", "--n", "6000", "--out", str(csv)], check=True)
    from dealerlens.data import load_raw
    from dealerlens.train import train
    df, _ = clean(load_raw(csv))
    results = train(df, out_dir=tmp / "artifacts", reports_dir=tmp / "reports", quick=True, log=lambda *a: None)
    from dealerlens.model import Appraiser
    return Appraiser(tmp / "artifacts"), results, tmp


def test_model_beats_baseline(trained):
    _, results, _ = trained
    assert results["lightgbm"]["mae"] < results["baseline_median_make_model_year"]["mae"]


def test_interval_contains_point(trained):
    app, _, _ = trained
    v, lo, hi = app.predict([{"year": 2016, "make": "toyota", "model": "camry", "odometer": 90000}])
    assert lo[0] <= v[0] <= hi[0]


def test_value_never_rises_with_age_and_miles(trained):
    from dealerlens.depreciation import curve
    app, _, _ = trained
    pts = curve(app, {"year": 2018, "make": "ford", "model": "F-150", "odometer": 40000}, years=range(6))
    vals = [p["value"] for p in pts]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_unknown_model_warns(trained):
    app, _, _ = trained
    _, warnings = app.normalize({"year": 2016, "make": "toyota", "model": "supra", "odometer": 50000})
    assert warnings


def test_appraise_and_api(trained, monkeypatch):
    app, _, tmp = trained
    from dealerlens.appraise import appraise
    monkeypatch.setattr(nhtsa, "_get", lambda url, params=None: {"results": []})
    nhtsa.recalls.cache_clear(); nhtsa.complaints.cache_clear()
    out = appraise(app, car={"year": 2016, "make": "honda", "model": "civic", "odometer": 70000}, price=9000)
    assert {"value", "low", "high", "factors", "depreciation", "deal", "risk"} <= out.keys()

    import api
    from fastapi.testclient import TestClient
    monkeypatch.setattr(api, "appraiser", lambda: app)
    from dealerlens.comps import Comps
    monkeypatch.setattr(api, "comps", lambda: Comps(tmp / "artifacts"))
    client = TestClient(api.app)
    ok = client.post("/appraise", json={"year": 2016, "make": "honda", "model": "civic", "odometer": 70000})
    assert ok.status_code == 200 and ok.json()["value"] > 0
    bad = client.post("/appraise", json={"make": "honda", "odometer": 70000})
    assert bad.status_code == 400
    assert ok.json()["comps"]["count"] > 0
    inv = "year,make,model,odometer,list_price\n2018,honda,civic,60000,15000\n"
    up = client.post("/inventory", files={"file": ("inv.csv", inv, "text/csv")})
    assert up.status_code == 200 and up.json()["summary"]["vehicles"] == 1
    from dealerlens import llm
    def down(*a, **k):
        raise llm.LLMUnavailable("offline")
    monkeypatch.setattr(llm, "chat", down)
    assert client.post("/copilot", json={"question": "hi"}).status_code == 503
    assert client.post("/memo", json={"year": 2016, "make": "honda", "model": "civic", "odometer": 70000,
                                      "with_risk": False}).json()["source"] == "template"


# ---------- listing text ----------
from dealerlens.text import build_text, scrub  # noqa: E402


@pytest.mark.parametrize("raw,must_not_contain,must_contain", [
    ("Asking $12,500 OBO", ["12,500", "12500"], ["num"]),
    ("price 9500 firm, 2015 camry", ["9500"], ["2015"]),
    ("$8k cash only", ["8k"], ["num"]),
    ("4wd v6, 2 owners, 85k miles", [], ["4wd", "v6", "2 owners"]),
])
def test_scrub_removes_prices_keeps_years(raw, must_not_contain, must_contain):
    out = scrub(raw)
    for s in must_not_contain:
        assert s not in out
    for s in must_contain:
        assert s in out


def test_build_text_skips_empty_parts():
    assert build_text("2015 Toyota Camry", None, "  ") == "2015 Toyota Camry"


def test_text_does_not_leak_price(trained):
    # Synthetic descriptions contain "Asking $<price>". If scrubbing failed, text alone would be near-perfect.
    _, results, _ = trained
    assert results["text_only_ridge"]["within_10pct"] < 0.6


def test_private_and_retail_values(trained, monkeypatch):
    app, _, _ = trained
    from dealerlens.appraise import appraise
    out = appraise(app, car={"year": 2018, "make": "toyota", "model": "camry", "odometer": 50000,
                             "notes": "one owner, new tires, runs great"}, with_risk=False)
    assert out["dealer_retail"]["value"] > out["private_party"]["value"]  # synthetic dealers list 12% higher
    assert out["text_signals"]["raises_value"]
    bad = appraise(app, car={"year": 2018, "make": "toyota", "model": "camry", "odometer": 50000,
                             "notes": "needs transmission, mechanic special"}, with_risk=False)
    assert bad["value"] < out["value"]


# ---------- Mondrian conformal ----------
def test_mondrian_fixes_under_covered_band():
    from dealerlens.metrics import band_index, mondrian_q
    rng = np.random.default_rng(1)
    pred = np.log(rng.choice([5_000, 50_000], size=40_000))
    noise = np.where(pred > np.log(20_000), 0.6, 0.2)          # expensive cars are noisier
    y = pred + rng.normal(0, 1, len(pred)) * noise
    lo, hi = pred - 0.2, pred + 0.2
    edges = [10_000, 20_000, 35_000]
    qs, gq = mondrian_q(lo[:20000], hi[:20000], y[:20000], pred[:20000], edges, alpha=0.2)
    b = band_index(pred[20000:], edges)
    q_row = np.asarray(qs)[b]
    yt, lt, ht = y[20000:], lo[20000:], hi[20000:]
    exp_mask = b == 3
    cov_global = ((yt >= lt - gq) & (yt <= ht + gq))[exp_mask].mean()
    cov_mondrian = ((yt >= lt - q_row) & (yt <= ht + q_row))[exp_mask].mean()
    assert cov_global < 0.75 and 0.77 <= cov_mondrian <= 0.83


# ---------- memo grounding ----------
def test_grounding_check():
    from dealerlens.memo import grounding_check
    facts = {"value": 18754, "deal": {"max_price_for_target_margin": 16454}}
    assert grounding_check("Retail $18,754, offer up to $16.5k", facts) == []      # rounding allowed
    assert grounding_check("Worth $25,000", facts) == ["$25,000"]


def test_memo_falls_back_without_llm(trained, monkeypatch):
    from dealerlens import llm
    from dealerlens.appraise import appraise
    from dealerlens.memo import llm_memo
    app, _, _ = trained

    def down(*a, **k):
        raise llm.LLMUnavailable("offline")
    monkeypatch.setattr(llm, "chat", down)
    r = appraise(app, car={"year": 2018, "make": "toyota", "model": "camry", "odometer": 50000}, price=12000, with_risk=False)
    m = llm_memo(r)
    assert m["source"] == "template" and "Recommendation" in m["memo"]


def test_memo_rejects_invented_numbers(trained, monkeypatch):
    from dealerlens import llm
    from dealerlens.appraise import appraise
    from dealerlens.memo import llm_memo
    app, _, _ = trained
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "Buy it. Worth $99,999.")
    r = appraise(app, car={"year": 2018, "make": "toyota", "model": "camry", "odometer": 50000}, with_risk=False)
    assert llm_memo(r)["source"] == "template"


# ---------- comps + inventory ----------
def test_comps(trained):
    from dealerlens.comps import Comps
    _, _, tmp = trained
    res = Comps(tmp / "artifacts").find({"year": 2018, "make": "honda", "model": "civic", "odometer": 60000}, k=5)
    assert res["count"] == 5 and all(abs(it["year"] - 2018) <= 2 for it in res["items"])
    assert Comps(tmp / "artifacts").find({"year": 2018, "make": "lada", "model": "niva", "odometer": 1})["count"] == 0


def test_inventory(trained):
    from dealerlens.batch import appraise_inventory, summarize
    app, _, _ = trained
    inv = pd.DataFrame({"stock": ["A1", "A2"], "year": [2018, 2015], "make": ["honda", "ford"],
                        "model": ["civic", "f-150"], "odometer": [60000, 120000],
                        "list_price": [99999, 1500], "cost": [9000, 1000]})
    out = appraise_inventory(app, inv)
    assert list(out["pricing_flag"]) == ["Above market: likely slow to sell", "Below market: leaving money on the table"]
    assert summarize(out)["vehicles"] == 2
    with pytest.raises(ValueError):
        appraise_inventory(app, inv.drop(columns=["odometer"]))


# ---------- agent ----------
def test_copilot_tool_loop(trained, monkeypatch):
    from dealerlens import llm
    from dealerlens.agent import Copilot
    from dealerlens.comps import Comps
    app, _, tmp = trained
    monkeypatch.setattr(nhtsa, "_get", lambda url, params=None: {"results": []})
    nhtsa.recalls.cache_clear(); nhtsa.complaints.cache_clear()
    car = '"year": 2018, "make": "honda", "model": "civic", "odometer": 60000'
    script = iter([
        '{"tool": "appraise_vehicle", "args": {' + car + ', "price": 9000}}',
        '```json\n{"tool": "find_comparables", "args": {' + car + '}}\n```',
        '{"tool": "check_safety", "args": {"year": 2018, "make": "honda", "model": "civic"}}',
        None,  # final: filled in below from the appraisal
    ])
    seen = {}

    def fake_chat(messages, **k):
        nxt = next(script)
        if nxt is None:
            first = [m for m in messages if m["content"].startswith("Tool result (appraise_vehicle)")][0]
            val = json.loads(first["content"].split("\n", 1)[1])["value"]
            seen["val"] = val
            return json.dumps({"final": f"Buy. Retail about ${val:,}. Also worth $77,777."})
        return nxt
    import json
    monkeypatch.setattr(llm, "chat", fake_chat)
    res = Copilot(app, Comps(tmp / "artifacts"), reports_dir=tmp / "reports").ask("2018 civic 60k miles for $9000?")
    assert [t["tool"] for t in res["trace"]] == ["appraise_vehicle", "find_comparables", "check_safety"]
    assert f"${seen['val']:,}" in res["answer"]
    assert res["unverified"] == ["$77,777."] or res["unverified"] == ["$77,777"]


def test_copilot_recovers_from_bad_tool(trained, monkeypatch):
    from dealerlens import llm
    from dealerlens.agent import Copilot
    app, _, tmp = trained
    script = iter(['{"tool": "teleport", "args": {}}', '{"final": "Sorry, I need the mileage."}'])
    monkeypatch.setattr(llm, "chat", lambda messages, **k: next(script))
    res = Copilot(app, None, reports_dir=tmp / "reports").ask("what's my car worth?")
    assert "error" in res["trace"][0]["result"] and "mileage" in res["answer"]


def test_notify_noop_without_webhook(monkeypatch):
    from dealerlens.notify import post_to_slack
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    assert post_to_slack("hi")["sent"] is False
