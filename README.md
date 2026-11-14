# DealerLens

**An appraisal copilot for used-car dealers.** Value a car in seconds with a price range you can trust, see real comparable listings and safety recalls, ask questions in plain English, and reprice a whole lot from a CSV. Runs fully on a laptop with a local LLM (no API keys, vehicle data stays on the machine).

| | |
|---|---|
| **Appraise** | Dealer-retail and private-party values, each with a calibrated 80% range · dollar impact of each factor · 3-year depreciation · Buy / Negotiate / Pass for an acquisition price |
| **Condition from text** | Paste inspection notes; see which phrases move the price ("needs transmission", "one owner") |
| **Comps** | Nearest real listings for the same model, year and mileage, with prices and links |
| **Risk** | NHTSA VIN decode, recall campaigns and complaint volume (free government APIs) |
| **Copilot (agent)** | Plain-English questions → the LLM plans and calls tools (valuation, comps, NHTSA, market trends) → a short buy memo. Every dollar amount is checked against tool output. |
| **Inventory** | Upload a lot CSV (DMS export) → market value per car, "above / below market" flags, expected gross |
| **Integrations** | REST API (FastAPI) · Slack webhook · memo download · Docker Compose |

## Results (held-out test set: 22,835 listings posted Sep-Nov 2026)

| Model | MAE | MAPE | Median APE | Within ±20% |
|---|---|---|---|---|
| Median by make + model + year (baseline) | $4,099 | 33.3% | 22.5% | 45.3% |
| Ridge regression (baseline) | $3,292 | 23.5% | 16.4% | 58.5% |
| Listing text only (TF-IDF ridge) | $3,645 | 26.0% | 18.7% | 52.9% |
| **LightGBM + text signal** | **$2,233** | **16.8%** | **11.8%** | **71.8%** |

- **45.5% lower MAE** than the make/model/year baseline. Typical dealer inventory ($10k-35k): **12.9% MAPE**.
- **80% price range coverage:** raw quantile models 72.0% → one global conformal widening 80.1% → **per price band (Mondrian) 80.1%**.
- Coverage by **predicted** price band (what a dealer sees at appraisal time): <10k 79%, 10-20k 81%, 20-35k 80%, 35k+ 80%.
- Coverage by **true** price band: <10k 80%, 10-20k 82%, 20-35k 80%, 35k+ 74%. Expensive cars tend to be under-predicted (estimates shrink toward the average), so coverage grouped by the true price is lower for $35k+; no method can guarantee coverage conditional on the unknown true price.
- **Iteration history:** v1 26.5% MAPE → v2 20.0% (correct data year, fake-price filter, trim) → v3 16.8% (listing text, seller type, odometer flags). v1→v2 also changed the cleaned test set, so part of that gain is data quality.

## How it works
```
VIN ──► NHTSA vPIC ──► features ──► LightGBM point model (log price; monotone in age & miles)
                                  ├─► LightGBM 10th/90th quantiles ─► Mondrian conformal ─► 80% range
                                  ├─► SHAP contributions ─► $ impact per factor
                                  └─► age + mileage sweep ─► depreciation curve
title + description / notes ─► scrub prices ─► TF-IDF + ridge (out-of-fold) ─► text score feature
make/model/year ─► NHTSA recalls + complaints ─► risk flags
cleaned listings ─► nearest-neighbor comps
Copilot: question ─► local LLM (Qwen2.5-7B, MLX) ⇄ tools ─► memo ─► dollar-amount grounding check
```
- **Leak-free evaluation:** train / valid / calibration / test split grouped by VIN; text model scored out-of-fold; dollar amounts scrubbed from listing text; `monthly_payment` excluded (it encodes the price).
- **Honest baselines:** make/model/year median and ridge regression.
- **Calibrated uncertainty:** conformal prediction per predicted-price band, so the "80% range" really covers ~80%, including expensive cars.
- **Grounded LLM:** the model never invents numbers. Memos with unverified amounts fall back to a template; the copilot flags them.

## Quickstart (Mac)
```bash
brew install libomp                     # LightGBM needs OpenMP on macOS
uv sync --group dev
uv run pytest                           # 41 tests, no data or LLM needed

# Data: Craigslist used-vehicle listings → data/vehicles.csv
uv run python -m scripts.eda --csv data/vehicles.csv
uv run python -m scripts.train --csv data/vehicles.csv
uv run python -m scripts.update_readme

# Local LLM for the copilot and AI memos (separate terminal; any OpenAI-compatible endpoint works)
mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8081

uv run streamlit run app.py             # web app
uv run uvicorn api:app --reload         # API docs at http://localhost:8000/docs
```

**Other LLM backends:** set `DEALERLENS_LLM_URL`, `DEALERLENS_LLM_MODEL`, `DEALERLENS_LLM_KEY` (e.g. Ollama at `http://localhost:11434/v1`).
**Slack:** set `SLACK_WEBHOOK_URL` to enable "Send to Slack".
**Docker:** `docker compose up --build` (LLM stays on the host).

## API
| Endpoint | What it does |
|---|---|
| `POST /appraise` | Full appraisal (+ comps, NHTSA) for a VIN or year/make/model/mileage |
| `POST /comps` | Comparable listings |
| `POST /memo` | Appraisal memo (LLM with grounding check, template fallback) |
| `POST /copilot` | Agent answer + tool trace + unverified amounts |
| `POST /inventory` | Upload a CSV → per-car values and pricing flags (`?format=csv` to download) |
| `POST /notify/slack` | Post text to Slack |

## Project layout
```
dealerlens/  data · text · train · model · metrics · depreciation · comps · deal · appraise ·
             nhtsa · batch · memo · agent · llm · notify · config
scripts/     make_sample · eda · train · appraise_cli · update_readme
api.py · app.py · Dockerfile · docker-compose.yml · MODEL_CARD.md · tests/
```

## Limitations
See [MODEL_CARD.md](MODEL_CARD.md): asking prices (not transactions), one snapshot in time (Sep-Nov 2026), self-reported condition, no vehicle-history data, NHTSA recalls by model rather than by VIN.
