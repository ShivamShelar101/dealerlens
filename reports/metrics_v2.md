# Results (held-out test set)

Rows: {'train': 96578, 'valid': 10707, 'calib': 22963, 'test': 22968}

| Model | MAE | MAPE | Median APE | Within ±10% | Within ±20% |
|---|---|---|---|---|---|
| Median by make + model + year | $3,998 | 32.9% | 22.7% | 24.2% | 44.9% |
| Ridge regression (log price) | $3,276 | 24.0% | 17.1% | 30.6% | 56.6% |
| **LightGBM (monotone in age & miles)** | $2,597 | 20.0% | 13.9% | 38.1% | 65.2% |

LightGBM MAE is **35.0% lower** than the make/model/year median baseline.

## 80% price range

| Method | Coverage (target 80%) | Mean width |
|---|---|---|
| Raw quantile models | 72.8% | $7,105 |
| **Conformal-calibrated** | 79.6% | $8,192 |

## By segment

| segment      | group     |     n |   mape |   within_10pct |   coverage_80 |
|:-------------|:----------|------:|-------:|---------------:|--------------:|
| price_band   | <10k      | 10268 | 0.2572 |         0.306  |        0.7886 |
| price_band   | 10-20k    |  7319 | 0.1544 |         0.4404 |        0.8155 |
| price_band   | 20-35k    |  3867 | 0.141  |         0.4738 |        0.798  |
| price_band   | 35k+      |  1514 | 0.1837 |         0.3593 |        0.7503 |
| age_band     | 0-3y      |  2107 | 0.1364 |         0.4945 |        0.804  |
| age_band     | 4-7y      |  4214 | 0.1548 |         0.4654 |        0.8073 |
| age_band     | 8-12y     |  7982 | 0.1824 |         0.4023 |        0.8011 |
| age_band     | 13y+      |  8665 | 0.2538 |         0.2916 |        0.7845 |
| mileage_band | <50k      |  2965 | 0.1802 |         0.4381 |        0.797  |
| mileage_band | 50-100k   |  6203 | 0.1759 |         0.4324 |        0.8036 |
| mileage_band | 100-150k  |  7802 | 0.1994 |         0.3772 |        0.8002 |
| mileage_band | 150k+     |  5998 | 0.2356 |         0.3029 |        0.7831 |
| type         | hatchback |   962 | 0.2044 |         0.3836 |        0.7713 |
| type         | other     |  3779 | 0.2266 |         0.3406 |        0.7764 |
| type         | pickup    |  3170 | 0.1901 |         0.3779 |        0.8082 |
| type         | sedan     |  5084 | 0.2078 |         0.3661 |        0.7966 |
| type         | suv       |  7868 | 0.1748 |         0.4245 |        0.804  |
| type         | truck     |  1135 | 0.2136 |         0.3665 |        0.8211 |
| type         | unknown   |   970 | 0.2722 |         0.2784 |        0.7649 |
