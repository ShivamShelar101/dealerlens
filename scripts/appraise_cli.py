"""Appraise from the command line.

    uv run python -m scripts.appraise_cli --vin 1HGCV1F34LA012345 --odometer 42000 --price 18500
    uv run python -m scripts.appraise_cli --year 2016 --make toyota --model camry --odometer 90000
"""
import argparse
import json

from dealerlens.appraise import appraise
from dealerlens.model import Appraiser

p = argparse.ArgumentParser()
p.add_argument("--vin")
p.add_argument("--year", type=int)
p.add_argument("--make")
p.add_argument("--model")
p.add_argument("--odometer", type=float, required=True)
p.add_argument("--condition", default="good")
p.add_argument("--title-status", default="clean")
p.add_argument("--state")
p.add_argument("--price", type=float, help="acquisition/asking price for a deal score")
p.add_argument("--recon", type=float, default=0, help="estimated reconditioning cost")
p.add_argument("--no-risk", action="store_true", help="skip NHTSA lookups")
a = p.parse_args()

car = {"year": a.year, "make": a.make, "model": a.model, "odometer": a.odometer,
       "condition": a.condition, "title_status": a.title_status, "state": a.state}
print(json.dumps(appraise(Appraiser(), car=car, vin=a.vin, price=a.price, recon=a.recon,
                          with_risk=not a.no_risk), indent=1, default=str))
