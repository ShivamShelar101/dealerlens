DATA_YEAR = 2026            # Listings were posted Sep-Nov 2026. Vehicle age = DATA_YEAR - model year.
ARTIFACTS = "artifacts"     # trained models + metadata
REPORTS = "reports"         # metrics, segments, leaderboard
ALPHA = 0.2                 # 1 - ALPHA = interval coverage target (80%)
BAND_EDGES = [10_000, 20_000, 35_000]  # Mondrian conformal: calibrate each predicted-price band separately

NUM_COLS = ["age", "odometer", "miles_per_year", "text_score"]
CAT_COLS = ["make", "model_key", "trim_key", "condition", "title_status", "cylinders",
            "fuel", "transmission", "drive", "type", "state", "seller_type", "paint_color"]
FEATURES = NUM_COLS + CAT_COLS

# Value can only go down as age and mileage go up (monotone constraints on the point model).
MONOTONE = [-1 if c in ("age", "odometer") else 0 for c in FEATURES]

LABELS = {
    "age": "Age", "odometer": "Mileage", "miles_per_year": "Miles per year", "text_score": "Listing text / notes",
    "make": "Make", "model_key": "Model", "trim_key": "Trim", "condition": "Condition",
    "title_status": "Title", "cylinders": "Cylinders", "fuel": "Fuel", "transmission": "Transmission",
    "drive": "Drivetrain", "type": "Body type", "state": "State", "seller_type": "Seller",
    "paint_color": "Color",
}
