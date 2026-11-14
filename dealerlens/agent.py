"""DealerLens Copilot: a tool-using agent for dealer questions in plain English.

The LLM plans and writes; every number comes from tools (the valuation model, comps, NHTSA).
Protocol is plain JSON, so it works with small local models and any OpenAI-compatible endpoint.
"""
import json
import pathlib

import pandas as pd

from dealerlens import config, llm
from dealerlens.appraise import appraise
from dealerlens.memo import grounding_check
from dealerlens.nhtsa import decode_vin, risk

CAR_ARGS = "year, make, model, odometer (required); trim, condition, title_status, state, notes (optional)"

TOOLS = {
    "decode_vin": "Decode a 17-character VIN into year/make/model/trim/specs. args: {vin}",
    "appraise_vehicle": f"Dealer-retail and private-party value with 80% ranges, top factors, 3-year depreciation, "
                        f"and a deal score if price is given. args: {CAR_ARGS}; price, recon (optional)",
    "find_comparables": f"Real comparable listings for the same model with their prices. args: {CAR_ARGS}",
    "check_safety": "NHTSA recalls and complaint volume. args: {year, make, model}",
    "market_trends": "Which models hold value best/worst over 3 years. args: {make} (optional)",
}

SYSTEM = f"""You are DealerLens Copilot, an assistant for a used-car dealer's buying team.

Tools:
{chr(10).join(f'- {k}: {v}' for k, v in TOOLS.items())}

How to respond:
- To call a tool, reply with ONLY a JSON object: {{"tool": "<name>", "args": {{...}}}}
- When you have what you need, reply with ONLY: {{"final": "<answer in markdown>"}}
- Call ONE tool per reply. You will get the result back.

Rules:
- Never invent numbers. Every dollar amount in your final answer must come from a tool result.
- For any question about a specific vehicle: call appraise_vehicle and find_comparables first; call check_safety too.
- If the user gives a price they'd pay, pass it as "price" to appraise_vehicle and give a Buy / Negotiate / Pass verdict.
- If mileage or year is missing and there is no VIN, ask for it in your final answer instead of guessing.
- Final answers: short memo, recommendation first, 3-5 bullets, then risks. Under 200 words.

Example:
User: 2018 Honda Accord EX, 70k miles, seller wants $15,500
You: {{"tool": "appraise_vehicle", "args": {{"year": 2018, "make": "honda", "model": "accord", "trim": "ex", "odometer": 70000, "price": 15500}}}}"""


def _parse(text):
    """First JSON object in the model's reply (tolerates code fences and chatter)."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except ValueError:
                continue
    return None


class Copilot:
    def __init__(self, app, comps=None, reports_dir=config.REPORTS, max_steps=7):
        self.app, self.comps, self.max_steps = app, comps, max_steps
        lb = pathlib.Path(reports_dir) / "leaderboard.csv"
        self.leaderboard = pd.read_csv(lb) if lb.exists() else None

    # ---------- tools ----------
    def _car(self, a):
        return {k: a[k] for k in ("year", "make", "model", "trim", "odometer", "condition", "title_status",
                                  "state", "notes") if a.get(k) not in (None, "")}

    def run_tool(self, name, a):
        a = a or {}
        if name == "decode_vin":
            return decode_vin(str(a["vin"]))
        if name == "appraise_vehicle":
            r = appraise(self.app, car=self._car(a), price=a.get("price"), recon=a.get("recon") or 0, with_risk=False)
            keep = ("value", "low", "high", "dealer_retail", "private_party", "retail_spread", "deal",
                    "warnings", "text_signals")
            out = {k: r[k] for k in keep if k in r}
            out["top_factors"] = r["factors"][:4]
            out["in_3_years"] = r["depreciation"][-1]
            return out
        if name == "find_comparables":
            if self.comps is None:
                return {"error": "comparables table not loaded"}
            res = self.comps.find(self._car(a), k=6)
            for it in res.get("items", []):
                it.pop("url", None)
                it.pop("title", None)
            return res
        if name == "check_safety":
            r = risk({"year": a.get("year"), "make": a.get("make"), "model": a.get("model")})
            return {"flags": r.get("flags", []), "recall_count": r.get("recalls", {}).get("count"),
                    "complaint_count": r.get("complaints", {}).get("count")}
        if name == "market_trends":
            if self.leaderboard is None:
                return {"error": "leaderboard not available"}
            lb = self.leaderboard
            if a.get("make"):
                lb = lb[lb["make"] == str(a["make"]).lower()]
            return {"holds_value_best": lb.head(5).to_dict("records"), "holds_value_worst": lb.tail(5).to_dict("records")}
        return {"error": f"unknown tool '{name}'. Available: {', '.join(TOOLS)}"}

    # ---------- loop ----------
    def ask(self, question, history=None):
        """Returns {"answer", "trace", "unverified", "steps"}. Raises llm.LLMUnavailable if no model."""
        messages = [{"role": "system", "content": SYSTEM}] + (history or []) + [{"role": "user", "content": question}]
        trace, reminded = [], False
        for step in range(self.max_steps):
            reply = llm.chat(messages, max_tokens=600)
            obj = _parse(reply)
            messages.append({"role": "assistant", "content": reply})
            if obj and "tool" in obj:
                try:
                    result = self.run_tool(obj["tool"], obj.get("args"))
                except Exception as e:  # bad args, NHTSA down, etc.: tell the model and let it recover
                    result = {"error": f"{type(e).__name__}: {e}"}
                trace.append({"tool": obj["tool"], "args": obj.get("args"), "result": result})
                messages.append({"role": "user", "content": f"Tool result ({obj['tool']}):\n{json.dumps(result, default=str)[:6000]}"})
                continue
            answer = obj["final"] if obj and "final" in obj else reply
            called = {t["tool"] for t in trace}
            if not reminded and "appraise_vehicle" not in called and any(ch.isdigit() for ch in question):
                reminded = True
                messages.append({"role": "user", "content": "Call appraise_vehicle (and find_comparables) before answering."})
                continue
            facts = [t["result"] for t in trace]
            return {"answer": answer, "trace": trace, "unverified": grounding_check(answer, facts), "steps": step + 1}
        return {"answer": "I couldn't finish within the step limit. Try a more specific question.",
                "trace": trace, "unverified": [], "steps": self.max_steps}
