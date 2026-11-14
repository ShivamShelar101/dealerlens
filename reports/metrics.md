# Results (held-out test set)

Rows: {'train': 95751, 'valid': 10644, 'calib': 22801, 'test': 22835}

| Model | MAE | MAPE | Median APE | Within ±10% | Within ±20% |
|---|---|---|---|---|---|
| Median by make + model + year | $4,099 | 33.3% | 22.5% | 23.8% | 45.3% |
| Ridge regression (log price) | $3,292 | 23.5% | 16.4% | 32.3% | 58.5% |
| Listing text only (TF-IDF ridge) | $3,645 | 26.0% | 18.7% | 28.4% | 52.9% |
| **LightGBM + text signal (monotone in age & miles)** | $2,233 | 16.8% | 11.8% | 43.7% | 71.8% |

LightGBM MAE is **45.5% lower** than the make/model/year median baseline.

## 80% price range

| Method | Coverage (target 80%) | Mean width |
|---|---|---|
| Raw quantile models | 72.0% | $6,303 |
| Conformal, one global widening | 80.1% | $7,439 |
| **Conformal per price band (Mondrian)** | 80.1% | $7,452 |

## By segment

| segment        | group     |     n |   mape |   within_10pct |   coverage_80 |
|:---------------|:----------|------:|-------:|---------------:|--------------:|
| price_band     | <10k      | 10024 | 0.216  |         0.3545 |        0.7966 |
| price_band     | 10-20k    |  7344 | 0.1336 |         0.491  |        0.8231 |
| price_band     | 20-35k    |  3827 | 0.1209 |         0.5383 |        0.7959 |
| price_band     | 35k+      |  1640 | 0.1428 |         0.4683 |        0.7439 |
| predicted_band | <10k      |  9811 | 0.2076 |         0.3498 |        0.7937 |
| predicted_band | 10-20k    |  7433 | 0.1448 |         0.4878 |        0.8139 |
| predicted_band | 20-35k    |  4020 | 0.1251 |         0.5413 |        0.7953 |
| predicted_band | 35k+      |  1571 | 0.1451 |         0.4799 |        0.8033 |
| age_band       | 0-3y      |  2194 | 0.1139 |         0.5775 |        0.8081 |
| age_band       | 4-7y      |  4282 | 0.133  |         0.5173 |        0.7978 |
| age_band       | 8-12y     |  7806 | 0.1521 |         0.4617 |        0.8036 |
| age_band       | 13y+      |  8553 | 0.2147 |         0.3393 |        0.799  |
| mileage_band   | <50k      |  3014 | 0.1449 |         0.5133 |        0.7999 |
| mileage_band   | 50-100k   |  6190 | 0.1508 |         0.4861 |        0.7985 |
| mileage_band   | 100-150k  |  7783 | 0.1659 |         0.4379 |        0.8039 |
| mileage_band   | 150k+     |  5848 | 0.2022 |         0.3461 |        0.8011 |
| type           | hatchback |   969 | 0.1773 |         0.4252 |        0.8029 |
| type           | other     |  3777 | 0.1819 |         0.4077 |        0.8004 |
| type           | pickup    |  3142 | 0.1606 |         0.437  |        0.81   |
| type           | sedan     |  4912 | 0.1747 |         0.4285 |        0.7966 |
| type           | suv       |  7900 | 0.1484 |         0.4778 |        0.8065 |
| type           | truck     |  1148 | 0.1805 |         0.3894 |        0.7997 |
| type           | unknown   |   987 | 0.2448 |         0.3404 |        0.7579 |
