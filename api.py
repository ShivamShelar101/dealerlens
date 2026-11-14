"""DealerLens API.

    uv run uvicorn api:app --reload
    curl -X POST localhost:8000/appraise -H 'Content-Type: application/json' \
      -d '{"year": 2016, "make": "toyota", "model": "camry", "odometer": 90000, "price": 11000}'
"""
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dealerlens.appraise import appraise
from dealerlens.model import Appraiser

app = FastAPI(title="DealerLens", description="Used-vehicle valuation with calibrated price ranges")


@lru_cache(maxsize=1)
def appraiser():
    return Appraiser()


class AppraiseRequest(BaseModel):
    vin: Optional[str] = Field(None, description="17-character VIN; fills year/make/model/specs")
    year: Optional[int] = None
    make: Optional[str] = None
    model: Optional[str] = None
    trim: Optional[str] = None
    notes: Optional[str] = Field(None, description="inspection notes or listing description")
    seller_type: Optional[str] = Field(None, description="leave empty to get both private-party and dealer-retail values")
    paint_color: Optional[str] = None
    odometer: float = Field(..., gt=0)
    condition: Optional[str] = "good"
    title_status: Optional[str] = "clean"
    state: Optional[str] = None
    drive: Optional[str] = None
    fuel: Optional[str] = None
    type: Optional[str] = None
    cylinders: Optional[str] = None
    transmission: Optional[str] = None
    price: Optional[float] = Field(None, gt=0, description="acquisition/asking price for a deal score")
    recon: float = Field(0, ge=0, description="estimated reconditioning cost")
    with_risk: bool = True


@app.get("/health")
def health():
    m = appraiser().meta
    return {"status": "ok", "data_year": m["data_year"], "test_metrics": m["metrics"]["lightgbm"],
            "interval_coverage": m["metrics"]["interval_conformal"]["coverage"]}


@app.post("/appraise")
def appraise_endpoint(req: AppraiseRequest):
    car = req.model_dump(exclude={"vin", "price", "recon", "with_risk"})
    try:
        return appraise(appraiser(), car=car, vin=req.vin, price=req.price, recon=req.recon,
                        with_risk=req.with_risk)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
