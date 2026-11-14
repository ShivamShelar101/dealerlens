"""DealerLens web app.

    uv run streamlit run app.py
"""
import pandas as pd
import streamlit as st

from dealerlens.appraise import appraise
from dealerlens.config import DATA_YEAR
from dealerlens.model import Appraiser

st.set_page_config(page_title="DealerLens", page_icon="🚗", layout="wide")


@st.cache_resource
def get_app():
    return Appraiser()


app = get_app()
m = app.meta["metrics"]
vocab = app.vocab
examples = pd.read_csv("artifacts/examples.csv")

st.title("🚗 DealerLens")
st.caption(f"Fair value with a calibrated 80% range · test MAPE {m['lightgbm']['mape']:.1%} · "
           f"range coverage {m['interval_conformal']['coverage']:.0%} · trained on listings posted in {DATA_YEAR}")

# ---------- Input ----------
if "car" not in st.session_state:
    st.session_state.car = {"year": 2016, "make": "toyota", "model": "camry", "odometer": 90000,
                            "condition": "good", "title_status": "clean", "state": "ca"}
if st.sidebar.button("🎲 Random car from the test set"):
    r = examples.sample(1).iloc[0]
    st.session_state.car = {"year": int(r["year"]), "make": r["make"], "model": r["model_key"],
                            "trim": "" if r["trim_key"] in ("base", "other") else r["trim_key"],
                            "odometer": int(r["odometer"]), "condition": r["condition"],
                            "title_status": r["title_status"], "state": r["state"],
                            "drive": r["drive"], "type": r["type"], "fuel": r["fuel"],
                            "cylinders": r["cylinders"], "transmission": r["transmission"],
                            "paint_color": r["paint_color"], "seller_type": r["seller_type"],
                            "notes": str(r.get("text", ""))[:600],
                            "_listed_price": int(r["price"])}
car0 = st.session_state.car

with st.sidebar:
    st.header("Vehicle")
    vin = st.text_input("VIN (optional, decodes year/make/model/specs via NHTSA)")
    makes = sorted(vocab)
    make = st.selectbox("Make", makes, index=makes.index(car0["make"]) if car0["make"] in makes else 0)
    models = [x for x in vocab.get(make, []) if x != "other"] + ["other"]
    model = st.selectbox("Model", models, index=models.index(car0["model"]) if car0["model"] in models else 0)
    trim = st.text_input("Trim (optional, e.g. XLT, Limited)", car0.get("trim", ""))
    year = st.number_input("Model year", 2000, DATA_YEAR + 1, int(car0["year"]))
    odometer = st.number_input("Mileage", 0, 400_000, int(car0["odometer"]), step=1000)
    conds = ["excellent", "like new", "good", "fair", "salvage", "unknown"]
    condition = st.selectbox("Condition", conds, index=conds.index(car0.get("condition", "good")) if car0.get("condition") in conds else 2)
    titles = ["clean", "rebuilt", "salvage", "lien", "missing", "parts only"]
    title = st.selectbox("Title", titles, index=titles.index(car0.get("title_status", "clean")) if car0.get("title_status") in titles else 0)
    state = st.text_input("State (2 letters)", car0.get("state", "ca"))
    colors = ["unknown"] + [c for c in app.categories.get("paint_color", []) if c != "unknown"]
    paint = st.selectbox("Color", colors, index=colors.index(car0.get("paint_color")) if car0.get("paint_color") in colors else 0)
    notes = st.text_area("Inspection notes / listing text (optional)", car0.get("notes", ""),
                         placeholder="e.g. one owner, new tires, check engine light on, small dent rear bumper")
    st.header("Deal")
    price = st.number_input("Acquisition / asking price ($, optional)", 0, 500_000, 0, step=500)
    recon = st.number_input("Estimated recon cost ($)", 0, 50_000, 0, step=100)
    with_risk = st.checkbox("NHTSA recall & complaint check", True)
    go = st.button("Appraise", type="primary")

if not go and "result" not in st.session_state:
    st.info("Enter a vehicle in the sidebar (or a VIN) and click **Appraise**.")
    st.stop()

if go:
    car = {**{k: v for k, v in car0.items() if not k.startswith("_")},
           "year": year, "make": make, "model": model, "trim": trim, "odometer": odometer,
           "paint_color": paint, "notes": notes,
           "condition": condition, "title_status": title, "state": state.lower()}
    if vin:  # let the VIN decode fill year/make/model/specs
        for k in ("year", "make", "model", "trim", "drive", "type", "fuel", "cylinders", "transmission"):
            car.pop(k, None)
    try:
        with st.spinner("Appraising..."):
            st.session_state.result = appraise(app, car=car, vin=vin or None, price=price or None,
                                               recon=recon, with_risk=with_risk)
    except Exception as e:
        st.error(str(e))
        st.stop()

r = st.session_state.result
c = r["car"]
st.subheader(f"{c.get('year')} {str(c.get('make', '')).title()} {str(c.get('model', '')).title()} · {float(c['odometer']):,.0f} mi")
for w in r["warnings"]:
    st.warning(w)

col1, col2, col3 = st.columns(3)
if "private_party" in r:
    pp, dr = r["private_party"], r["dealer_retail"]
    col1.metric("Private-party value", f"${pp['value']:,}", help=f"80% range ${pp['low']:,} – ${pp['high']:,}")
    col2.metric("Dealer retail value", f"${dr['value']:,}", f"+${r['retail_spread']:,} vs private",
                help=f"80% range ${dr['low']:,} – ${dr['high']:,}")
    st.caption(f"80% ranges: private ${pp['low']:,} – ${pp['high']:,} · retail ${dr['low']:,} – ${dr['high']:,}")
else:
    col1.metric("Fair market value", f"${r['value']:,}")
    col2.metric("80% range", f"${r['low']:,} – ${r['high']:,}")
in3 = r["depreciation"][-1]
col3.metric("Projected value in 3 years", f"${in3['value']:,}", f"-{100 - in3['retained_pct']:.0f}%", delta_color="inverse")
if car0.get("_listed_price") and not vin:
    st.caption(f"Actual listed price for this test-set car: ${car0['_listed_price']:,}")

if r.get("text_signals"):
    ts = r["text_signals"]
    st.markdown("**From the notes:** " +
                (" ".join(f":green[+ {p}]" for p in ts["raises_value"]) + " " +
                 " ".join(f":red[− {p}]" for p in ts["lowers_value"])).strip())

if "deal" in r:
    d = r["deal"]
    color = {"Buy": "green", "Negotiate": "orange", "Pass": "red"}[d["label"]]
    st.markdown(f"### Deal score: :{color}[{d['label']}]  (vs. dealer retail value)" if "dealer_retail" in r
                else f"### Deal score: :{color}[{d['label']}]")
    st.write(f"Expected margin **{d['expected_margin_pct']}%** on total cost ${d['total_cost']:,} · "
             f"probability market value exceeds cost **{d['prob_value_above_cost']:.0%}** · "
             f"max price for an 8% margin: **${d['max_price_for_target_margin']:,}**")

left, right = st.columns(2)
with left:
    st.markdown("#### Why this value")
    f = pd.DataFrame(r["factors"])
    f["label"] = f["feature"] + ": " + f["value"].astype(str)
    st.bar_chart(f.set_index("label")["impact_usd"], horizontal=True)
    st.caption("Approximate $ effect of each factor versus an average car in the data.")
with right:
    st.markdown("#### Depreciation outlook")
    dep = pd.DataFrame(r["depreciation"]).set_index("years_ahead")[["low", "value", "high"]]
    st.line_chart(dep)
    st.caption("Assumes the car keeps its current miles per year. Market-wide price changes are not modeled.")

if "risk" in r:
    st.markdown("#### NHTSA risk check")
    rk = r["risk"]
    if rk.get("flags"):
        for fl in rk["flags"]:
            st.error(fl)
    elif rk.get("note"):
        st.caption(rk["note"])
    else:
        st.success("No recall or high-complaint flags.")
    if rk.get("recalls", {}).get("items"):
        with st.expander(f"Recalls ({rk['recalls']['count']})"):
            for it in rk["recalls"]["items"]:
                st.markdown(f"**{it['campaign']}** · {it['component']}  \n{it['summary']}")

with st.expander("Model performance (held-out test set)"):
    st.markdown(open("reports/metrics.md").read())
with st.expander("Which models hold value best (typical 3-year-old example, next 3 years)"):
    st.dataframe(pd.read_csv("reports/leaderboard.csv"), hide_index=True)
