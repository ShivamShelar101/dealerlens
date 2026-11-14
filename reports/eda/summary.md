# EDA summary

## Cleaning funnel

| Rule | Rows left |
|---|---|
| raw | 295,451 |
| price_1k_150k | 279,843 |
| year_2000_plus | 219,985 |
| odometer_500_400k | 216,410 |
| has_make_model | 208,921 |
| plausible_miles_per_year | 184,255 |
| not_new_and_cheap | 183,967 |
| deduplicated | 122,641 |

Kept 41.5% of raw rows.

## Top 15 make/model

| Make / model | Listings |
|---|---|
| ford f150 | 3,952 |
| chevrolet silverado | 3,543 |
| jeep wrangler | 2,522 |
| ram 1500 | 2,275 |
| honda accord | 1,871 |
| toyota camry | 1,830 |
| honda civic | 1,745 |
| ford escape | 1,668 |
| ford f250 | 1,619 |
| honda crv | 1,562 |
| toyota tacoma | 1,544 |
| ford explorer | 1,541 |
| jeep grand cherokee | 1,508 |
| subaru forester | 1,456 |
| gmc sierra | 1,427 |

## Clean price

count    122641.0
mean      11783.0
std        9768.0
min        1000.0
25%        5500.0
50%        9000.0
75%       14999.0
max      150000.0
