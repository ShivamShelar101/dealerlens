# Model card: DealerLens valuation model

## Intended use
Estimate the fair market (listing) value of a used passenger vehicle in the US, with an 80% price range, to support dealer acquisition and pricing decisions. A decision aid, not an automated pricing system.

## Data
- **Source:** Craigslist used-vehicle listings scraped across the US (Austin Reese dataset, Kaggle), listings posted September-November 2026.
- **Cleaning:** see `reports/eda/summary.md` for the funnel. Kept listings priced $1k–$150k, model year 2000+, 500–400k miles, plausible miles per year, not "new and cheap" (usually lease or down-payment ads), deduplicated reposts, and dropped prices more than 3x away from the median of the same make/model/year (fake or placeholder prices).
- **Rows:** _fill in from `artifacts/meta.json`_ (train / valid / calibration / test).

## Model
- LightGBM regression on log price, monotone decreasing in age and mileage.
- LightGBM quantile models (10th, 90th percentile) → conformal calibration on a separate split → 80% range. Calibration is done per predicted-price band (Mondrian conformal: <$10k, $10-20k, $20-35k, $35k+) so expensive cars aren't under-covered.
- Splits grouped by VIN.
- Listing-text score: TF-IDF (1-2 grams) + ridge on title + description, prices scrubbed, out-of-fold on train; at inference it scores a generated title plus the dealer's notes.
- Seller type (dealer vs private owner) is a feature, so the model reports both a private-party and a dealer-retail value.
- Excluded on purpose: `monthly_payment` (≈ price / 60, leaks the target), image counts and URLs (listing artifacts).

## Results (held-out test set)
_Paste `reports/metrics.md`._

## Where it's weaker
_Fill in from `reports/segments.csv`: which price bands, ages, mileage bands and body types have the highest error or lowest coverage._

## Limitations
- **Asking prices, not transaction prices.** Dealer acquisition and auction prices are typically lower.
- **One snapshot in time.** Listings were posted Sep-Nov 2026, so the model has no notion of seasonal or market-wide price trends; it needs periodic retraining.
- **Self-reported fields.** Condition and title status come from sellers.
- **No vehicle history.** Accidents, number of owners and service records aren't in the data; those are exactly what a vehicle-history product would add.
- **Trim-level detail is coarse.** Models are normalized to a base name (e.g., "silverado"), so trims are pooled.
- **Text at inference differs from training.** Training text is full listings (often dealer boilerplate); at inference it's a short title plus optional notes. The text effect is most reliable for clear condition phrases.
- **NHTSA flags are heuristics.** Recall counts are by make/model/year, not by VIN, and don't show whether a recall was completed.

## Next steps
- Retrain on recent auction / dealer transaction data
- Add vehicle-history features (accidents, owners, service)
- Trim-level matching from VIN decode
- Monitor drift and interval coverage monthly

## LLM components (copilot and memos)
- The LLM never produces numbers on its own: values, ranges, comps and recalls come from tools.
- Every dollar amount in an LLM answer is checked against tool output (within 1% or $50). Memos that fail the
  check are replaced by a template; copilot answers show unverified amounts as a warning.
- Default model is a local 4-bit Qwen2.5-7B via MLX, so vehicle data never leaves the machine.
