"""Fixed project constants for the AgriSence crop price forecasting scope.

The final market and crop scope was determined by coverage analysis in
``01_eda.ipynb``. Maize and Maize (White), and Beans and Beans (Dry), are
merged after unit normalization.
"""

MARKETS = ["Nairobi", "Kisumu", "Mombasa", "Nakuru"]
CROPS = ["Maize", "Beans"]
HORIZONS = ["1_month", "2_month"]

TOTAL_COMBINATIONS = len(MARKETS) * len(CROPS) * len(HORIZONS)
