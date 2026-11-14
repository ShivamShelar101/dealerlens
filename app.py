"""DealerLens web app.

    uv run streamlit run app.py
"""
import io

import pandas as pd
import streamlit as st

from dealerlens import llm
from dealerlens.agent import Copilot
from dealerlens.appraise import appraise
from dealerlens.batch import appraise_inventory, summarize
from dealerlens.comps import Comps
from dealerlens.config import DATA_YEAR
from dealerlens.memo import llm_memo, template_memo
from dealerlens.model import Appraiser
from dealerlens.notify import post_to_slack, slack_enabled

st.set_page_config(page_title="DealerLens", page_icon="🚗", layout="wide")


@st.cache_resource
def load():
    return Appraiser(), Comps()


app, comps = load()
m = app.meta["metrics"]
vocab = app.vocab
examples = pd.read_csv("artifacts/examples.csv")

st.title("🚗 DealerLens")
st.caption(f"Appraisal copilot for used-car dealers · test MAPE {m['lightgbm']['mape']:.1%} · "
           f"80% range coverage {m['interval_conformal']['coverage']:.0%} · trained on listings posted in {DATA_YEAR}")

# =====================================================================  Sidebar: vehicle
if "car" not in st.session_state:
    st.session_state.car = {"year": 2018, "make": "toyota", "model": "camry", "odometer": 70000,
                            "condition": "good", "title_status": "clean", "state": "ca"}
if st.sidebar.button("🎲 Random car from the test set"):
    r = examples.sample(1).iloc[0]
    st.session_state.car = {"year": int(r["year"]), "make": r["make"], "model": r["model_key"],
                            "trim": "" if r["trim_key"] in ("base", "other") else r["trim_key"],
                            "odometer": int(r["odometer"]), "condition": r["condition"],
                            "title_status": r["title_status"], "state": r["state"],
                            "drive": r["drive"], "type": r["type"], "fuel": r["fuel"],
                            "cylinders": r["cylinders"], "transmission": r["transmission"],
                            "paint_color": r["paint_color"], "notes": str(r.get("text", ""))[:600],
                            "_listed_price": int(r["price"]), "_id": r.get("id")}
car0 = st.session_state.car

with st.sidebar:
    st.header("Vehicle")
    vin = st.text_input("VIN (optional, decoded via NHTSA)")
    makes = sorted(vocab)
    make = st.selectbox("Make", makes, index=makes.index(car0["make"]) if car0["make"] in makes else 0)
    models = [x for x in vocab.get(make, []) if x != "other"] + ["other"]
    model = st.selectbox("Model", models, index=models.index(car0["model"]) if car0["model"] in models else 0)
    trim = st.text_input("Trim (optional, e.g. XLT, Limited)", car0.get("trim", ""))
    year = st.number_input("Model year", 2000, DATA_YEAR + 1, int(car0["year"]))
    odometer = st.number_input("Mileage", 0, 400_000, int(car0["odometer"]), step=1000)
    conds = ["excellent", "like new", "good", "fair", "salvage", "unknown"]
    condition = st.selectbox("Condition", conds, index=conds.index(car0.get("condition")) if car0.get("condition") in conds else 2)
    titles = ["clean", "rebuilt", "salvage", "lien", "missing", "parts only"]
    title = st.selectbox("Title", titles, index=titles.index(car0.get("title_status")) if car0.get("title_status") in titles else 0)
    state = st.text_input("State (2 letters)", car0.get("state", "ca"))
    colors = ["unknown"] + [c for c in app.categories.get("paint_color", []) if c != "unknown"]
    paint = st.selectbox("Color", colors, index=colors.index(car0.get("paint_color")) if car0.get("paint_color") in colors else 0)
    notes = st.text_area("Inspection notes / listing text", car0.get("notes", ""),
                         placeholder="e.g. one owner, new tires, check engine light on, dent rear bumper")
    st.header("Deal")
    price = st.number_input("Acquisition price ($, optional)", 0, 500_000, 0, step=500)
    recon = st.number_input("Estimated recon cost ($)", 0, 50_000, 0, step=100)
    with_risk = st.checkbox("NHTSA recall & complaint check", True)
    go = st.button("Appraise", type="primary", use_container_width=True)

tab_appraise, tab_copilot, tab_inventory, tab_market = st.tabs(["🔎 Appraise", "🤖 Copilot", "📋 Inventory", "📈 Market"])

# =====================================================================  Appraise
with tab_appraise:
    if go:
        car = {**{k: v for k, v in car0.items() if not k.startswith("_")},
               "year": year, "make": make, "model": model, "trim": trim, "odometer": odometer,
               "paint_color": paint, "notes": notes,
               "condition": condition, "title_status": title, "state": state.lower()}
        if vin:
            for k in ("year", "make", "model", "trim", "drive", "type", "fuel", "cylinders", "transmission"):
                car.pop(k, None)
        try:
            with st.spinner("Appraising..."):
                r = appraise(app, car=car, vin=vin or None, price=price or None, recon=recon, with_risk=with_risk)
                r["comps"] = comps.find(r["car"], exclude_id=car0.get("_id"))
                st.session_state.result, st.session_state.memo = r, None
        except Exception as e:
            st.error(str(e))

    if "result" not in st.session_state:
        st.info("Enter a vehicle in the sidebar (or a VIN) and click **Appraise**. "
                "Or ask the **Copilot** in plain English, or upload your lot in **Inventory**.")
    else:
        r = st.session_state.result
        c = r["car"]
        st.subheader(f"{c.get('year')} {str(c.get('make', '')).title()} {str(c.get('model', '')).title()} "
                     f"{str(c.get('trim') or '').upper()} · {float(c['odometer']):,.0f} mi")
        for w in r["warnings"]:
            st.warning(w)

        c1, c2, c3 = st.columns(3)
        if "private_party" in r:
            pp, dr = r["private_party"], r["dealer_retail"]
            c1.metric("Private-party value", f"${pp['value']:,}")
            c2.metric("Dealer retail value", f"${dr['value']:,}", f"+${r['retail_spread']:,} vs private")
            st.caption(f"80% ranges · private ${pp['low']:,} – ${pp['high']:,} · retail ${dr['low']:,} – ${dr['high']:,}")
        else:
            c1.metric("Market value", f"${r['value']:,}")
            c2.metric("80% range", f"${r['low']:,} – ${r['high']:,}")
        in3 = r["depreciation"][-1]
        c3.metric("Value in 3 years", f"${in3['value']:,}", f"-{100 - in3['retained_pct']:.0f}%", delta_color="inverse")
        if car0.get("_listed_price") and not vin:
            st.caption(f"Actual listed price of this test-set car: ${car0['_listed_price']:,}")

        if r.get("text_signals"):
            ts = r["text_signals"]
            st.markdown("**From the notes:** " + " ".join(f":green[+ {p}]" for p in ts["raises_value"]) + " "
                        + " ".join(f":red[− {p}]" for p in ts["lowers_value"]))
        if "deal" in r:
            d = r["deal"]
            color = {"Buy": "green", "Negotiate": "orange", "Pass": "red"}[d["label"]]
            st.markdown(f"### Deal score: :{color}[{d['label']}]")
            st.write(f"Expected margin **{d['expected_margin_pct']}%** on total cost ${d['total_cost']:,} · "
                     f"probability market value exceeds cost **{d['prob_value_above_cost']:.0%}** · "
                     f"max price for an 8% margin: **${d['max_price_for_target_margin']:,}**")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Why this value")
            f = pd.DataFrame(r["factors"])
            f["label"] = f["feature"] + ": " + f["value"].astype(str)
            st.bar_chart(f.set_index("label")["impact_usd"], horizontal=True)
            st.caption("Approximate $ effect of each factor versus an average listing.")
        with right:
            st.markdown("#### Depreciation outlook")
            st.line_chart(pd.DataFrame(r["depreciation"]).set_index("years_ahead")[["low", "value", "high"]])
            st.caption("Assumes current miles per year. Market-wide price trends are not modeled.")

        cp = r.get("comps", {})
        st.markdown(f"#### Comparable listings ({cp.get('count', 0)})")
        if cp.get("count"):
            st.caption(f"Median ${cp['median_price']:,} · range ${cp['price_range'][0]:,} – ${cp['price_range'][1]:,}")
            cdf = pd.DataFrame(cp["items"])
            st.dataframe(cdf, hide_index=True, use_container_width=True,
                         column_config={"url": st.column_config.LinkColumn("listing"),
                                        "price": st.column_config.NumberColumn(format="$%d")})
        else:
            st.caption(cp.get("note", "No comparables."))

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

        st.markdown("#### Appraisal memo")
        b1, b2 = st.columns(2)
        if b1.button("✍️ Write memo with AI"):
            with st.spinner("Writing memo with the local model..."):
                st.session_state.memo = llm_memo(r, r.get("comps"), r.get("risk"))
        if b2.button("📄 Quick memo (no AI)"):
            st.session_state.memo = {"memo": template_memo(r, r.get("comps"), r.get("risk")), "source": "template"}
        memo = st.session_state.get("memo")
        if memo:
            if memo.get("note"):
                st.caption(f"ℹ️ {memo['note']}")
            st.markdown(memo["memo"])
            st.caption("Written by the local LLM; every $ amount verified against model output." if memo["source"] == "llm"
                       else "Generated from model output (no LLM).")
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Download memo (.md)", memo["memo"], file_name="appraisal_memo.md")
            if slack_enabled() and d2.button("💬 Send to Slack"):
                st.toast("Sent to Slack" if post_to_slack(memo["memo"]).get("sent") else "Slack send failed")

    with st.expander("Model performance (held-out test set)"):
        st.markdown(open("reports/metrics.md").read())

# =====================================================================  Copilot
with tab_copilot:
    st.markdown("Ask in plain English. The copilot calls the valuation model, comparable listings, NHTSA and "
                "market trends as tools, and **every dollar figure is checked against tool output**.")
    if not llm.available():
        st.warning(f"No LLM reachable at `{llm.URL}`. Start the local model server, e.g. "
                   "`mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8081`.")
    st.caption("Try: *2017 Ford F-150 XLT 4x4, 92k miles, small dent rear bumper, seller wants $19,500. Should we buy?*  ·  "
               "*Which trucks hold value best?*")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for turn in st.session_state.chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("trace"):
                with st.expander(f"🔧 {len(turn['trace'])} tool call(s)"):
                    for t in turn["trace"]:
                        st.markdown(f"**{t['tool']}** `{t['args']}`")
                        st.json(t["result"], expanded=False)
            if turn.get("unverified"):
                st.warning(f"Unverified amounts (not found in tool output): {', '.join(turn['unverified'])}")
    q = st.chat_input("Ask about a vehicle, a deal, or the market")
    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        history = [{"role": t["role"], "content": t["content"]} for t in st.session_state.chat[:-1]][-6:]
        try:
            with st.spinner("Thinking and calling tools..."):
                res = Copilot(app, comps).ask(q, history)
            st.session_state.chat.append({"role": "assistant", "content": res["answer"], "trace": res["trace"],
                                          "unverified": res["unverified"]})
        except llm.LLMUnavailable as e:
            st.session_state.chat.append({"role": "assistant", "content": f"⚠️ {e}"})
        st.rerun()

# =====================================================================  Inventory
with tab_inventory:
    st.markdown("Upload your lot (a CSV export from your DMS or a spreadsheet). Required columns: "
                "`year, make, model, odometer`. Optional: `stock, vin, trim, condition, title_status, state, "
                "notes, list_price, cost`.")
    sample = examples.head(25).rename(columns={"model_key": "model_name"})
    demo = pd.DataFrame({"stock": [f"A{100 + i}" for i in range(len(sample))], "year": sample["year"],
                         "make": sample["make"], "model": sample["model_name"], "odometer": sample["odometer"],
                         "condition": sample["condition"], "state": sample["state"],
                         "list_price": (sample["price"] * 1.05).round(-2), "cost": (sample["price"] * 0.8).round(-2)})
    st.download_button("⬇️ Download a sample inventory CSV", demo.to_csv(index=False), file_name="sample_inventory.csv")
    up = st.file_uploader("Inventory CSV", type=["csv"])
    if up:
        try:
            res = appraise_inventory(app, pd.read_csv(up))
            s = summarize(res)
            k1, k2, k3 = st.columns(3)
            k1.metric("Vehicles", s["vehicles"])
            k2.metric("Total market retail", f"${s['total_market_retail']:,}")
            if "total_expected_gross" in s:
                k3.metric("Expected gross (retail − cost)", f"${s['total_expected_gross']:,}")
            if "flags" in s:
                st.write({k: v for k, v in s["flags"].items()})
            st.dataframe(res, hide_index=True, use_container_width=True)
            buf = io.StringIO()
            res.to_csv(buf, index=False)
            st.download_button("⬇️ Download repriced inventory", buf.getvalue(), file_name="inventory_appraised.csv")
            if slack_enabled() and st.button("💬 Send summary to Slack"):
                post_to_slack(f"DealerLens inventory check: {s}")
                st.toast("Sent")
        except ValueError as e:
            st.error(str(e))

# =====================================================================  Market
with tab_market:
    st.markdown("#### Which models hold value best (typical 3-year-old example, next 3 years)")
    st.dataframe(pd.read_csv("reports/leaderboard.csv"), hide_index=True, use_container_width=True)
    st.markdown("#### Model accuracy by segment")
    st.dataframe(pd.read_csv("reports/segments.csv"), hide_index=True, use_container_width=True)
