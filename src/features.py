"""
Reusable feature engineering for the Mersey River discharge model.

Mirrors the steps in notebooks/02_feature_engineering.ipynb exactly, extracted here so both
that notebook AND the Streamlit app build features the same way. This avoids "train/serve skew" --
a common real-world bug where a model is trained on features computed one way, then served in
production with slightly different feature logic, silently degrading predictions.
"""

import numpy as np
import pandas as pd

# lag windows in days -- short lags capture day-to-day persistence, longer lags capture
# slower drawdown after a storm event
LAG_DAYS = [1, 2, 3, 7, 14, 30]

# rolling window sizes in days, for smoothed trend + volatility features
ROLLING_WINDOWS = [7, 30]

# the exact feature columns the trained model expects, in a fixed order -- the app must
# build a dataframe with these same column names before calling model.predict()
FEATURE_COLS = [
    "discharge_cms",
    "discharge_lag1", "discharge_lag2", "discharge_lag3",
    "discharge_lag7", "discharge_lag14", "discharge_lag30",
    "discharge_roll_mean_7", "discharge_roll_std_7",
    "discharge_roll_mean_30", "discharge_roll_std_30",
    "doy_sin", "doy_cos",
]


def build_features(df: pd.DataFrame, discharge_col: str = "discharge_cms") -> pd.DataFrame:
    """
    Build the full feature set from a daily discharge series.

    df : DataFrame with a DatetimeIndex and a discharge column (m3/s). Should already be
         reindexed to a continuous daily frequency with gaps filled -- this function assumes
         no missing dates, since lag/rolling logic depends on evenly spaced days.
    discharge_col : name of the discharge column in df.

    Returns a new DataFrame with all engineered feature columns added (does NOT drop rows
    with NaNs from the lag/rolling warmup period -- caller decides whether to dropna(),
    since the live app needs to keep the most recent rows even if some optional columns
    aren't fully populated yet).
    """
    out = df.copy()

    # lag features
    for lag in LAG_DAYS:
        out[f"discharge_lag{lag}"] = out[discharge_col].shift(lag)

    # rolling mean/std -- shifted by 1 first so the window only looks at PAST days,
    # never leaking the current day's own value into its own feature
    for window in ROLLING_WINDOWS:
        out[f"discharge_roll_mean_{window}"] = out[discharge_col].shift(1).rolling(window).mean()
        out[f"discharge_roll_std_{window}"] = out[discharge_col].shift(1).rolling(window).std()

    # cyclical day-of-year encoding, so Dec 31 and Jan 1 are treated as adjacent
    day_of_year = out.index.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    return out


def latest_feature_row(df: pd.DataFrame, discharge_col: str = "discharge_cms") -> pd.DataFrame:
    """
    Convenience wrapper for the app: builds features and returns just the most recent
    row, ready to hand to model.predict(). Raises if that row has any missing feature
    values (e.g. not enough history yet for a 30-day rolling window).
    """
    featured = build_features(df, discharge_col=discharge_col)
    latest = featured.iloc[[-1]][FEATURE_COLS]

    if latest.isna().any(axis=None):
        missing = latest.columns[latest.isna().iloc[0]].tolist()
        raise ValueError(
            f"Not enough history to compute all features for the latest row. "
            f"Missing: {missing}. Need at least {max(LAG_DAYS + ROLLING_WINDOWS)} days of "
            f"prior discharge data."
        )

    return latest
