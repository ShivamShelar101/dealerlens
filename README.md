# DealerLens

**Used-vehicle valuation for dealers, with a price range you can trust.** Enter a VIN (or year/make/model) and mileage and get:

- **Private-party and dealer-retail values**, each with a **calibrated 80% price range** (conformal prediction; coverage is measured on held-out data)
- **Condition from text**: paste inspection notes and see which phrases move the price ("needs transmission", "one owner")
- **Why**: the top factors with their approximate dollar impact
- **Depreciation outlook**: projected value in 1, 2 and 3 years
- **NHTSA risk check**: recall campaigns and owner-complaint volume (free government APIs)
- **Deal score** for an acquisition price: expected margin, probability of being underwater, Buy / Negotiate / Pass

FastAPI service + Streamlit app. Trained on ~120k cleaned US used-car listings posted September-November 2026.

## Results (held-out test set, 22,835 listings posted Sep-Nov 2026)

| Model | MAE | MAPE | Median APE | Within ±20% |
|---|---|---|---|---|
| Median by make + model + year (baseline) | $4,099 | 33.3% | 22.5% | 45.3% |
| Ridge regression (baseline) | $3,292 | 23.5% | 16.4% | 58.5% |
| Listing text only (TF-IDF ridge) | $3,645 | 26.0% | 18.7% | 52.9% |
| **LightGBM + text signal** | **$2,233** | **16.8%** | **11.8%** | **71.8%** |

- 45.5% lower MAE than the make/model/year baseline. For typical dealer inventory ($10k-35k): 12-13% MAPE.
- **80% price range:** raw quantile models covered 72.0% of true prices; after conformal calibration, **80.1%**.
- **Iteration history:** v1 26.5% MAPE -> v2 20.0% (correct data year, fake-price filter, trim) -> v3 16.8% (listing text, seller type, odometer flags). v1->v2 also changed the cleaned test set, so part of that gain is data quality, not modeling.
- **Known gap:** cars over $35k are under-covered (73.8% vs 80% target). Next step: per-price-band (Mondrian) conformal calibration.

## How it works
```
VIN ──► NHTSA vPIC decode ──► features ──► LightGBM point model (log price, monotone in age/miles)
                                         ├─► LightGBM 10th/90th quantile models ─► conformal calibration ─► 80% range
                                         ├─► built-in SHAP contributions ─► $ impact per factor
                                         └─► age + mileage sweep ─► depreciation curve
make / model / year ──► NHTSA recalls + complaints ─► risk flags
acquisition price + recon ──► deal score
```
- **Leak-free splits:** train / valid (early stopping) / calibration (conformal) / test, grouped by VIN so the same car never lands in two splits.
- **Monotone constraints:** the point model can never value a car higher because it's older or has more miles.
- **Honest baselines:** results are reported against a make/model/year median and a ridge regression.
- **Leak-safe text signal:** TF-IDF + ridge on listing title/description with dollar amounts and price-like numbers scrubbed, out-of-fold on the training set. `monthly_payment` is excluded because it encodes the price.

## Quickstart
```bash
uv sync --group dev
uv run pytest                                   # 20 tests, no data needed

# Dry run on synthetic data (1 minute)
uv run python -m scripts.make_sample --n 20000 --out data/sample.csv
uv run python -m scripts.train --csv data/sample.csv
uv run streamlit run app.py
```

## Real data
Download the Craigslist used-vehicles dataset (Austin Reese) from Kaggle (`austinreese/craigslist-carstrucks-data`) and put `vehicles.csv` in `data/`. Then:
```bash
uv run python -m scripts.eda --csv data/vehicles.csv      # plots + cleaning funnel → reports/eda/
uv run python -m scripts.train --csv data/vehicles.csv    # models → artifacts/, results → reports/
uv run streamlit run app.py
uv run uvicorn api:app --reload                            # API at localhost:8000/docs
uv run python -m scripts.appraise_cli --year 2016 --make toyota --model camry --odometer 90000 --price 11000
```

## Project layout
```
dealerlens/  data.py · train.py · model.py · metrics.py · depreciation.py · nhtsa.py · deal.py · appraise.py · config.py
scripts/     make_sample.py · eda.py · train.py · appraise_cli.py
api.py       FastAPI service      app.py   Streamlit app
reports/     metrics.md · segments.csv · leaderboard.csv · eda/
tests/       data rules, leakage, conformal coverage, deal score, NHTSA mapping (mocked), end-to-end + API
```

## Limitations
See [MODEL_CARD.md](MODEL_CARD.md). In short: listing (asking) prices, self-reported condition, a single snapshot (Sep-Nov 2026) with no price trend over time, and no vehicle-history signals (accidents, owners).
