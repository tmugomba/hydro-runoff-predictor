# Reusable logic for the hydro run-of-river predictor.
#
# features.py -- shared feature engineering (lags, rolling stats, seasonal encoding),
#                used by both notebooks/02_feature_engineering.ipynb and the future
#                Streamlit app, so training and serving never drift apart.
# power.py     -- discharge-to-power conversion, calibrated as a lumped equivalent
#                against the real 42.5 MW Mersey Hydro System.
