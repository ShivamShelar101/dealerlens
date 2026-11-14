"""DealerLens API.

    uv run uvicorn api:app --reload        # docs at http://localhost:8000/docs

Endpoints: /health · /appraise · /comps · /memo · /copilot · /inventory (CSV upload) · /notify/slack
"""
import io
from functools import lru_cache
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from dealerlens import llm
from dealerlens.agent import Copilot
from dealerlens.appraise import appraise
from dealerlens.batch import appraise_inventory, summarize
from dealerlens.comps import Comps
from dealerlens.memo import llm_memo
from dealerlens.model import Appraiser
from dealerlens.notify import post_to_slack

app = FastAPI(title="DealerLens", description="Used-vehicle appraisal copilot: calibrated values, comps, risk, agent")


@lru_cache(maxsize=1)
def appraiser():
    return Appraiser()


@lru_cache(maxsize=1)
def comps():
    return Comps()


class Vehicle(BaseModel):
    vin: Optional[str] = Field(None, description="17-character VIN; fills year/make/model/trim/specs")
    year: Optional[int] = None
    make: Optional[str] = None
    model: Optional[str] = None
    trim: Optional[str] = None
    odometer: float = Field(..., gt=0)
    condition: Optional[str] = "good"
    title_status: Optional[str] = "clean"
    state: Optional[str] = None
    drive: Optional[str] = None
    fuel: Optional[str] = None
    type: Optional[str] = None
    cylinders: Optional[str] = None
    transmission: Optional[str] = None
    paint_color: Optional[str] = None
    notes: Optional[str] = Field(None, description="inspection notes or listing description")
    seller_type: Optional[str] = Field(None, description="leave empty to get both private-party and dealer-retail values")


class AppraiseRequest(Vehicle):
    price: Optional[float] = Field(None, gt=0, description="acquisition/asking price for a deal score")
    recon: float = Field(0, ge=0, description="estimated reconditioning cost")
    with_risk: bool = True
    with_comps: bool = True


class CopilotRequest(BaseModel):
    question: str
    history: list = Field(default_factory=list, description="prior [{role, content}] turns")


class SlackRequest(BaseModel):
    text: str


def _appraise(req: AppraiseRequest):
    car = req.model_dump(exclude={"vin", "price", "recon", "with_risk", "with_comps"})
    try:
        out = appraise(appraiser(), car=car, vin=req.vin, price=req.price, recon=req.recon, with_risk=req.with_risk)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if req.with_comps:
        out["comps"] = comps().find(out["car"])
    return out


@app.get("/health")
def health():
    m = appraiser().meta
    return {"status": "ok", "data_year": m["data_year"], "test_metrics": m["metrics"]["lightgbm"],
            "interval_coverage": m["metrics"]["interval_conformal"]["coverage"], "llm_available": llm.available()}


@app.post("/appraise")
def appraise_endpoint(req: AppraiseRequest):
    return _appraise(req)


@app.post("/comps")
def comps_endpoint(req: Vehicle, k: int = 8):
    car = {k_: v for k_, v in req.model_dump().items() if v not in (None, "")}
    if not all(car.get(f) for f in ("year", "make", "model")):
        raise HTTPException(status_code=400, detail="year, make and model are required")
    return comps().find(car, k=k)


@app.post("/memo")
def memo_endpoint(req: AppraiseRequest):
    out = _appraise(req)
    return llm_memo(out, out.get("comps"), out.get("risk"))


@app.post("/copilot")
def copilot_endpoint(req: CopilotRequest):
    try:
        return Copilot(appraiser(), comps()).ask(req.question, req.history)
    except llm.LLMUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/inventory")
async def inventory_endpoint(file: UploadFile = File(...), format: str = "json"):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        result = appraise_inventory(appraiser(), df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if format == "csv":
        buf = io.StringIO()
        result.to_csv(buf, index=False)
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=inventory_appraised.csv"})
    return {"summary": summarize(result), "vehicles": result.where(result.notna(), None).to_dict("records")}


@app.post("/notify/slack")
def slack_endpoint(req: SlackRequest):
    return post_to_slack(req.text)
