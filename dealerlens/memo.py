"""Dealer-facing appraisal memos: a deterministic template, an LLM-written version, and a check that
every dollar figure the LLM writes actually came from the model/tool outputs."""
import json
import re

from dealerlens import llm

_DOLLARS = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s?(k|K)?")


def _numbers(obj, out=None):
    out = [] if out is None else out
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers(v, out)
    return out


def grounding_check(text, facts):
    """Dollar amounts in `text` that don't match any number in `facts` (within 1% or $50)."""
    known = _numbers(facts)
    unverified = []
    for m in _DOLLARS.finditer(text):
        val = float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)
        if not any(abs(val - k) <= max(50, 0.01 * abs(k)) for k in known):
            unverified.append(m.group(0).strip())
    return unverified


def _fmt(n):
    return f"${n:,.0f}"


def template_memo(appraisal, comps=None, safety=None):
    """No-LLM memo built directly from the numbers. Always available."""
    c = appraisal["car"]
    name = f"{c.get('year')} {str(c.get('make', '')).title()} {str(c.get('model', '')).title()} {c.get('trim') or ''}".strip()
    lines = [f"## Appraisal: {name}, {float(c['odometer']):,.0f} mi", ""]
    if "deal" in appraisal:
        d = appraisal["deal"]
        lines += [f"**Recommendation: {d['label']}.** Total cost {_fmt(d['total_cost'])}, expected margin "
                  f"{d['expected_margin_pct']}%, max price for an 8% margin {_fmt(d['max_price_for_target_margin'])}.", ""]
    if "private_party" in appraisal:
        pp, dr = appraisal["private_party"], appraisal["dealer_retail"]
        lines += [f"- **Dealer retail:** {_fmt(dr['value'])} (80% range {_fmt(dr['low'])}–{_fmt(dr['high'])})",
                  f"- **Private party:** {_fmt(pp['value'])} (80% range {_fmt(pp['low'])}–{_fmt(pp['high'])})"]
    else:
        lines += [f"- **Market value:** {_fmt(appraisal['value'])} (80% range {_fmt(appraisal['low'])}–{_fmt(appraisal['high'])})"]
    dep = appraisal["depreciation"][-1]
    lines += [f"- **In 3 years:** {_fmt(dep['value'])} ({dep['retained_pct']}% retained)", "", "**Main factors:**"]
    lines += [f"- {f['feature']} ({f['value']}): {'+' if f['impact_usd'] >= 0 else '−'}{_fmt(abs(f['impact_usd']))}"
              for f in appraisal["factors"][:4]]
    if appraisal.get("text_signals"):
        ts = appraisal["text_signals"]
        lines += ["", f"**From the notes:** raises value: {', '.join(ts['raises_value']) or 'none'}; "
                      f"lowers value: {', '.join(ts['lowers_value']) or 'none'}."]
    if comps and comps.get("count"):
        lines += ["", f"**Comparable listings:** {comps['count']} found, median {_fmt(comps['median_price'])} "
                      f"(range {_fmt(comps['price_range'][0])}–{_fmt(comps['price_range'][1])})."]
    if safety and safety.get("flags"):
        lines += ["", "**Risk flags:**"] + [f"- {f}" for f in safety["flags"]]
    for w in appraisal.get("warnings", []):
        lines += ["", f"_Note: {w}_"]
    return "\n".join(lines)


MEMO_PROMPT = """You write short appraisal memos for a used-car dealer's buying team.
Use ONLY the facts in the JSON. Every dollar amount you write must appear in the JSON (you may round to the nearest $100).
Structure: one-line recommendation (Buy / Negotiate / Pass, with a max offer if a deal score exists), then 3-5 bullets on why
(values and range, comparable listings, condition notes, depreciation), then risks (recalls, warnings, wide range).
Under 180 words. Markdown."""


def llm_memo(appraisal, comps=None, safety=None):
    """LLM-written memo with a grounding check; falls back to the template if the LLM is unavailable
    or writes numbers that aren't in the facts."""
    facts = {"appraisal": {k: v for k, v in appraisal.items() if k not in ("risk",)},
             "comparables": comps, "safety": safety}
    try:
        text = llm.chat([{"role": "system", "content": MEMO_PROMPT},
                         {"role": "user", "content": json.dumps(facts, default=str)[:12000]}], max_tokens=450)
    except llm.LLMUnavailable as e:
        return {"memo": template_memo(appraisal, comps, safety), "source": "template", "note": str(e)}
    bad = grounding_check(text, facts)
    if bad:
        return {"memo": template_memo(appraisal, comps, safety), "source": "template",
                "note": f"LLM memo rejected: unverified amounts {bad}"}
    return {"memo": text, "source": "llm", "unverified": []}
