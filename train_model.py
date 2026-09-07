"""
Run this once locally to (re)generate app/discharge_model.pkl using whatever XGBoost
version is actually installed in YOUR venv. This avoids the "input stream corrupted"
error, which happens when a saved XGBoost model's binary format doesn't match the
version of XGBoost trying to load it (or the .pkl got altered in transit through
zip/git on a different machine).

Usage (from the project root, with your venv activated):
    python train_model.py

Mirrors the exact training code from notebooks/03_modeling.ipynb -- same feature
columns, same split, same hyperparameters -- so results match what's documented in
the README.
"""

import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# same engineered feature table used everywhere else in the project
model_df = pd.read_csv("data/processed/mersey_features.csv", index_col="date", parse_dates=True)

# same 80/20 chronological split as the notebook -- no shuffling, since this is a
# time series and training on "future" data would leak information into the past
split_idx = int(len(model_df) * 0.8)
train_df = model_df.iloc[:split_idx]
test_df = model_df.iloc[split_idx:]

FEATURE_COLS = [
    "discharge_cms",
    "discharge_lag1", "discharge_lag2", "discharge_lag3",
    "discharge_lag7", "discharge_lag14", "discharge_lag30",
    "discharge_roll_mean_7", "discharge_roll_std_7",
    "discharge_roll_mean_30", "discharge_roll_std_30",
    "doy_sin", "doy_cos",
]
TARGET_COL = "target_discharge_next_day"

X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

# same conservative hyperparameters as the notebook, chosen to reduce overfitting
# risk on a single-station dataset
model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

# quick sanity-check print so you can confirm this matches the documented finding
# (baseline usually wins overall -- see README for why that's a real result, not a bug)
xgb_pred = model.predict(X_test)
baseline_pred = test_df["discharge_cms"]

print(f"Baseline -- MAE: {mean_absolute_error(y_test, baseline_pred):.2f}, "
      f"RMSE: {root_mean_squared_error(y_test, baseline_pred):.2f}, "
      f"R2: {r2_score(y_test, baseline_pred):.3f}")
print(f"XGBoost  -- MAE: {mean_absolute_error(y_test, xgb_pred):.2f}, "
      f"RMSE: {root_mean_squared_error(y_test, xgb_pred):.2f}, "
      f"R2: {r2_score(y_test, xgb_pred):.3f}")

joblib.dump(model, "app/discharge_model.pkl")
print("\nSaved app/discharge_model.pkl -- matches the XGBoost version in this venv.")
