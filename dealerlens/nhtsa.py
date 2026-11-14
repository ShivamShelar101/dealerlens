"""Free NHTSA APIs (no key): VIN decode (vPIC), recalls and complaints by make/model/year."""
import re
from collections import Counter
from functools import lru_cache

import requests

VPIC = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}?format=json"
RECALLS = "https://api.nhtsa.gov/recalls/recallsByVehicle"
COMPLAINTS = "https://api.nhtsa.gov/complaints/complaintsByVehicle"
VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")


def _get(url, params=None):
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def _drive(s):
    s = (s or "").lower()
    if any(k in s for k in ("4wd", "4x4", "4-wheel", "awd", "all-wheel")):
        return "4wd"
    if "front" in s or "fwd" in s:
        return "fwd"
    if "rear" in s or "rwd" in s:
        return "rwd"
    return "unknown"


def _fuel(s):
    s = (s or "").lower()
    for key, out in (("diesel", "diesel"), ("electric", "electric"), ("hybrid", "hybrid"), ("gasoline", "gas")):
        if key in s:
            return out
    return "other" if s else "unknown"


def _type(s):
    s = (s or "").lower()
    for key, out in (("pickup", "pickup"), ("sport utility", "suv"), ("suv", "suv"), ("minivan", "mini-van"),
                     ("van", "van"), ("convertible", "convertible"), ("coupe", "coupe"),
                     ("hatchback", "hatchback"), ("wagon", "wagon"), ("sedan", "sedan"), ("truck", "truck")):
        if key in s:
            return out
    return "unknown"


def _transmission(s):
    s = (s or "").lower()
    if "manual" in s:
        return "manual"
    if "automatic" in s or "cvt" in s:
        return "automatic"
    return "unknown"


def vpic_to_car(r):
    """Map a vPIC DecodeVinValues result to DealerLens fields."""
    cyl = (r.get("EngineCylinders") or "").strip()
    return {
        "year": int(r["ModelYear"]) if (r.get("ModelYear") or "").isdigit() else None,
        "make": (r.get("Make") or "").lower().strip() or None,
        "model": (r.get("Model") or "").strip() or None,
        "trim": (r.get("Trim") or "").strip() or None,
        "cylinders": f"{cyl} cylinders" if cyl.isdigit() else "unknown",
        "fuel": _fuel(r.get("FuelTypePrimary")),
        "drive": _drive(r.get("DriveType")),
        "type": _type(r.get("BodyClass")),
        "transmission": _transmission(r.get("TransmissionStyle")),
    }


@lru_cache(maxsize=2048)
def decode_vin(vin):
    vin = vin.strip().upper()
    if not VIN_RE.match(vin):
        raise ValueError("A VIN is 17 characters and never contains I, O or Q.")
    car = vpic_to_car(_get(VPIC.format(vin=vin))["Results"][0])
    if not car["make"] or not car["year"]:
        raise ValueError("NHTSA couldn't decode this VIN. Check it, or enter the car manually.")
    return car


@lru_cache(maxsize=2048)
def recalls(make, model, year):
    try:
        data = _get(RECALLS, {"make": make, "model": model, "modelYear": year})
    except Exception as e:
        return {"count": None, "items": [], "error": str(e)}
    res = data.get("results") or data.get("Results") or []
    return {"count": len(res), "items": [
        {"campaign": x.get("NHTSACampaignNumber"), "component": x.get("Component"),
         "summary": (x.get("Summary") or "")[:300]} for x in res[:10]]}


@lru_cache(maxsize=2048)
def complaints(make, model, year):
    try:
        data = _get(COMPLAINTS, {"make": make, "model": model, "modelYear": year})
    except Exception as e:
        return {"count": None, "top_components": [], "error": str(e)}
    res = data.get("results") or data.get("Results") or []
    comps = Counter(c.strip() for x in res for c in (x.get("components") or "").split(",") if c.strip())
    return {"count": len(res), "top_components": comps.most_common(3)}


def risk(car, recall_threshold=1, complaint_threshold=100):
    """Heuristic flags from NHTSA data. Thresholds are placeholders to tune with the team."""
    make, model, year = car.get("make"), car.get("model"), car.get("year")
    if not (make and model and year):
        return {"flags": [], "note": "Need make, model and year for NHTSA lookups."}
    rc, cp = recalls(make, model, int(year)), complaints(make, model, int(year))
    flags = []
    if rc["count"] and rc["count"] >= recall_threshold:
        flags.append(f"{rc['count']} recall campaign(s): verify they were completed.")
    if cp["count"] and cp["count"] >= complaint_threshold:
        top = ", ".join(c for c, _ in cp["top_components"])
        flags.append(f"High complaint volume ({cp['count']}), mostly: {top}.")
    return {"flags": flags, "recalls": rc, "complaints": cp}
