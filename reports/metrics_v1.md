# Results (held-out test set)

Rows: {'train': 77295, 'valid': 8565, 'calib': 18366, 'test': 18415}

| Model | MAE | MAPE | Within ±10% |
|---|---|---|---|
| Median by make + model + year | $3,794 | 43.2% | 21.1% |
| Ridge regression (log price) | $2,963 | 30.4% | 27.9% |
| **LightGBM (monotone in age & miles)** | $2,498 | 26.5% | 33.1% |

LightGBM MAE is **34.2% lower** than the make/model/year median baseline.

## 80% price range

| Method | Coverage (target 80%) | Mean width |
|---|---|---|
| Raw quantile models | 71.8% | $6,385 |
| **Conformal-calibrated** | 79.6% | $7,496 |

## By segment

| segment      | group     |     n |   mape |   within_10pct |   coverage_80 |
|:-------------|:----------|------:|-------:|---------------:|--------------:|
| price_band   | <10k      | 10348 | 0.3324 |         0.2911 |        0.7931 |
| price_band   | 10-20k    |  5757 | 0.1652 |         0.4087 |        0.8275 |
| price_band   | 20-35k    |  1790 | 0.1947 |         0.3341 |        0.7654 |
| price_band   | 35k+      |   520 | 0.2759 |         0.2385 |        0.6077 |
| age_band     | 0-3y      |  2794 | 0.1787 |         0.4137 |        0.8114 |
| age_band     | 4-7y      |  6731 | 0.2509 |         0.3736 |        0.8064 |
| age_band     | 8-12y     |  4735 | 0.2665 |         0.3039 |        0.7926 |
| age_band     | 13y+      |  4155 | 0.3447 |         0.2351 |        0.7723 |
| mileage_band | <50k      |  1089 | 0.3077 |         0.2883 |        0.7622 |
| mileage_band | 50-100k   |  4619 | 0.2317 |         0.3765 |        0.8039 |
| mileage_band | 100-150k  |  6897 | 0.2574 |         0.3475 |        0.8008 |
| mileage_band | 150k+     |  5810 | 0.2929 |         0.2818 |        0.7902 |
| type         | hatchback |   794 | 0.2558 |         0.3048 |        0.7897 |
| type         | other     |  3215 | 0.2775 |         0.3104 |        0.7841 |
| type         | pickup    |  2322 | 0.2828 |         0.3053 |        0.8019 |
| type         | sedan     |  4208 | 0.2716 |         0.3329 |        0.7868 |
| type         | suv       |  6126 | 0.2316 |         0.3645 |        0.8116 |
| type         | truck     |   880 | 0.3098 |         0.317  |        0.7795 |
| type         | unknown   |   870 | 0.341  |         0.2586 |        0.7793 |
