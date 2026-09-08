"""
Mersey River Run-of-River Discharge & Power Dashboard
------------------------------------------------------
Streamlit app for the hydro pillar of the portfolio. Pulls the historical WSC discharge
record for station 01ED003 (Mersey River at Milton, NS), engineers the same features used
in training (src/features.py), runs the saved XGBoost next-day forecast model, and converts
discharge to an estimated power output using the lumped-equivalent calibration against the
real 42.5 MW Mersey Hydro System (src/power.py).

Run locally:
    streamlit run app/app.py

Author: Tendekai Mugomba
LinkedIn: https://www.linkedin.com/in/tendekai-mugomba
GitHub:   https://github.com/tmugomba
"""

import sys
import os
import math

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

# make src/ importable regardless of the working directory streamlit is launched from
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, ROOT_DIR)

from src.data import fetch_wsc_discharge, STATION_ID, STATION_NAME, STATION_LAT, STATION_LON
from src.features import build_features, FEATURE_COLS
from src.power import (
    calibrate_power_constant,
    discharge_to_power,
    capacity_factor,
    MERSEY_SYSTEM_RATED_CAPACITY_MW,
    DESIGN_FLOW_QUANTILE,
)
from theme import CSS, tooltip, svg_img
from gauges import speedometer_svg
from schematic import schematic_svg

TARGET_COL = "target_discharge_next_day"
MODEL_PATH = os.path.join(APP_DIR, "discharge_model.pkl")
RAW_CSV_PATH = os.path.join(ROOT_DIR, "data", "raw", "mersey_daily_discharge_raw.csv")

st.set_page_config(
    page_title="Mersey River Hydro Dashboard",
    page_icon="\U0001F4A7",
    layout="wide",
)
# st.html() (not st.markdown) for raw CSS/HTML injection -- markdown parsing can
# occasionally mangle or literally display a <style> block instead of applying it;
# st.html() renders the string as-is, with no markdown interpretation in the way
st.html(CSS)


# ----------------------------------------------------------------------------------------
# Data / model loading (cached so the app doesn't refetch or retrain on every interaction)
# ----------------------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading discharge record...")
def load_discharge() -> pd.DataFrame:
    """Historical daily discharge for the station, via the shared fetch module (falls
    back to the local cached CSV automatically if the live WSC endpoint isn't reachable)."""
    return fetch_wsc_discharge(station=STATION_ID, local_fallback_path=RAW_CSV_PATH)


@st.cache_data(show_spinner="Engineering features...")
def load_feature_table(discharge_df: pd.DataFrame) -> pd.DataFrame:
    """Same feature set the model was trained on, plus the next-day target column, with
    warmup-period NaNs dropped (mirrors 02_feature_engineering.ipynb)."""
    featured = build_features(discharge_df, discharge_col="discharge_cms")
    featured[TARGET_COL] = featured["discharge_cms"].shift(-1)
    return featured.dropna()


@st.cache_resource(show_spinner="Loading forecast model...")
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Scoring model performance...")
def compute_performance(feature_df: pd.DataFrame):
    """Recomputes the single 80/20 split metrics AND the 5-fold TimeSeriesSplit robustness
    check live from the current feature table, so the numbers shown always match the data
    actually loaded (rather than being hardcoded from a past notebook run)."""
    split_idx = int(len(feature_df) * 0.8)
    train_df, test_df = feature_df.iloc[:split_idx], feature_df.iloc[split_idx:]
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    baseline_pred = test_df["discharge_cms"]
    model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X_train, y_train)
    xgb_pred = model.predict(X_test)

    single_split = pd.DataFrame({
        "model": ["Naive persistence", "XGBoost"],
        "MAE (m3/s)": [mean_absolute_error(y_test, baseline_pred), mean_absolute_error(y_test, xgb_pred)],
        "RMSE (m3/s)": [root_mean_squared_error(y_test, baseline_pred), root_mean_squared_error(y_test, xgb_pred)],
        "R2": [r2_score(y_test, baseline_pred), r2_score(y_test, xgb_pred)],
    })

    X_full, y_full = feature_df[FEATURE_COLS], feature_df[TARGET_COL]
    tscv = TimeSeriesSplit(n_splits=5)
    fold_rows = []
    for fold_num, (tr_idx, te_idx) in enumerate(tscv.split(X_full), start=1):
        X_tr, X_te = X_full.iloc[tr_idx], X_full.iloc[te_idx]
        y_tr, y_te = y_full.iloc[tr_idx], y_full.iloc[te_idx]
        base_pred = X_te["discharge_cms"]
        fold_model = XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
        fold_model.fit(X_tr, y_tr)
        fold_pred = fold_model.predict(X_te)
        fold_rows.append({
            "Fold": fold_num,
            "Test window": f"{X_te.index.min().date()} to {X_te.index.max().date()}",
            "Baseline MAE": mean_absolute_error(y_te, base_pred),
            "XGBoost MAE": mean_absolute_error(y_te, fold_pred),
            "XGBoost wins": mean_absolute_error(y_te, fold_pred) < mean_absolute_error(y_te, base_pred),
        })
    cv_df = pd.DataFrame(fold_rows)
    return single_split, cv_df


discharge_df = load_discharge()
feature_df = load_feature_table(discharge_df)
model = load_model()

design_flow = discharge_df["discharge_cms"].quantile(DESIGN_FLOW_QUANTILE)
K = calibrate_power_constant(design_flow_cms=design_flow)
full_power_series = discharge_to_power(discharge_df["discharge_cms"], k=K)
overall_cf = capacity_factor(full_power_series)


# ----------------------------------------------------------------------------------------
# Sidebar -- pick a historical date to inspect
# ----------------------------------------------------------------------------------------

st.sidebar.markdown("### Station")
st.sidebar.markdown(
    f"**{STATION_NAME}**  \nWSC `{STATION_ID}`  \n"
    f"Record: {discharge_df.index.min().date()} to {discharge_df.index.max().date()}"
)
st.sidebar.markdown(
    "This station was discontinued in 1979, so this dashboard treats its record as a "
    "**fixed historical dataset** for backtesting and analysis, not a live feed. Pick any "
    "day below to see that day's conditions and the model's next-day forecast."
)

min_date = feature_df.index.min().date()
max_date = feature_df.index.max().date()  # feature_df already excludes the very last day (no next-day target)

selected_date = st.sidebar.date_input(
    "Inspect a date",
    value=max_date,
    min_value=min_date,
    max_value=max_date,
)
selected_ts = pd.Timestamp(selected_date)
if selected_ts not in feature_df.index:
    st.sidebar.warning("No data for that exact date (gap in the record) -- showing the nearest available day.")
    selected_ts = feature_df.index[feature_df.index.get_indexer([selected_ts], method="nearest")[0]]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data source:** [WSC data availability]"
    f"(https://wateroffice.ec.gc.ca/report/data_availability_e.html?type=historical&station={STATION_ID}&parameter_type=Flow)"
)


# ----------------------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------------------

st.html(
    """
<div class="hero-block">
  <div class="hero-title">Mersey River Run-of-River Dashboard</div>
  <div class="hero-sub">Discharge forecasting &amp; power estimation for Nova Scotia's Mersey Hydro System, built on 25 years of WSC gauge data.</div>
</div>
"""
)


# ----------------------------------------------------------------------------------------
# Compute today / next-day values for the selected date
# ----------------------------------------------------------------------------------------

row = feature_df.loc[selected_ts]
today_discharge = row["discharge_cms"]
actual_next_day = row[TARGET_COL]
naive_forecast = today_discharge  # persistence baseline: tomorrow = today

X_row = pd.DataFrame([row[FEATURE_COLS].values], columns=FEATURE_COLS)
xgb_forecast = float(model.predict(X_row)[0])

today_power = float(discharge_to_power(today_discharge, k=K))
today_cf = today_power / MERSEY_SYSTEM_RATED_CAPACITY_MW
flow_percentile = (discharge_df["discharge_cms"] <= today_discharge).mean() * 100


# ----------------------------------------------------------------------------------------
# Gauges
# ----------------------------------------------------------------------------------------

st.html(f'<div class="section-label">Conditions on {selected_ts.date()}</div>')
g1, g2, g3 = st.columns(3)

discharge_scale_max = float(discharge_df["discharge_cms"].quantile(0.99))

# tooltip text pulled out into plain variables (rather than inlined in the f-strings
# below) so apostrophes/quotes in the explanations don't have to fight with Python's
# string-escaping rules inside an already-nested f-string expression
discharge_tip = (
    "How much water is flowing past the gauge station right now, in cubic metres "
    "per second (m3/s). More discharge means more water available to generate power."
)
power_tip = (
    "Today's discharge converted to an estimated electricity output in megawatts "
    "(MW), capped at the real system's 42.5 MW rated capacity -- see the power "
    "conversion note further down the page for how this conversion works."
)
capacity_factor_tip = (
    "Today's power output as a percentage of the plant's maximum possible output "
    "(42.5 MW) if it ran flat-out, all day, every day. Real run-of-river hydro "
    "typically sits around 40-60%, since flow varies with the season."
)

with g1:
    st.html(
        f'<div class="card gauge-card">'
        f'<div class="card-title" style="text-align:center;">DISCHARGE{tooltip(discharge_tip)}</div>'
        f'{svg_img(speedometer_svg(today_discharge, 0, discharge_scale_max, "DISCHARGE", "m3/s", color="#22d3ee"))}'
        f'</div>'
    )
with g2:
    st.html(
        f'<div class="card gauge-card">'
        f'<div class="card-title" style="text-align:center;">POWER OUTPUT{tooltip(power_tip)}</div>'
        f'{svg_img(speedometer_svg(today_power, 0, MERSEY_SYSTEM_RATED_CAPACITY_MW, "POWER OUTPUT", "MW", color="#f5a623"))}'
        f'</div>'
    )
with g3:
    st.html(
        f'<div class="card gauge-card">'
        f'<div class="card-title" style="text-align:center;">CAPACITY FACTOR{tooltip(capacity_factor_tip)}</div>'
        f'{svg_img(speedometer_svg(today_cf * 100, 0, 100, "CAPACITY FACTOR", "%", value_fmt="{:.0f}", color="#22d3ee"))}'
        f'</div>'
    )

percentile_tip = (
    f"Where today's flow ranks against every day in the full 1954-1979 record. "
    f"The {flow_percentile:.0f}th percentile means only {100 - flow_percentile:.0f}% "
    f"of days on record had higher flow than this."
)
avg_cf_tip = (
    "The average of the capacity factor gauge above, taken across the entire "
    "25-year historical record rather than just the selected day."
)
st.html(
    f'<div style="color:#8a93a6;font-size:0.85rem;margin-top:0.3rem;">'
    f'Today\'s flow sits at the {flow_percentile:.0f}th percentile{tooltip(percentile_tip)}'
    f' of the full 1954-1979 record &nbsp;|&nbsp; '
    f'whole-record average capacity factor: {overall_cf:.1%}{tooltip(avg_cf_tip)}'
    f'</div>'
)


# ----------------------------------------------------------------------------------------
# Animated schematic
# ----------------------------------------------------------------------------------------

st.html(f'<div class="schematic-wrap">{svg_img(schematic_svg(today_power, MERSEY_SYSTEM_RATED_CAPACITY_MW))}</div>')


# ----------------------------------------------------------------------------------------
# Next-day forecast
# ----------------------------------------------------------------------------------------

st.html('<div class="section-label">Next-day discharge forecast</div>')
f1, f2, f3 = st.columns(3)
with f1:
    st.html(
        f'<div class="card">'
        f'<div class="card-title">Naive baseline{tooltip("Tomorrow = today. River discharge is highly autocorrelated day-to-day, making this a surprisingly strong baseline.")}</div>'
        f'<span class="mono" style="font-size:1.7rem;color:#e8ecf3;">{naive_forecast:.1f}</span> <span style="color:#8a93a6;">m3/s</span>'
        f'</div>'
    )
with f2:
    st.html(
        f'<div class="card">'
        f'<div class="card-title">XGBoost forecast{tooltip("Gradient-boosted trees trained on lags, rolling stats, and seasonal encoding. See Model Performance below for when this does and doesn\'t beat the baseline.")}</div>'
        f'<span class="mono" style="font-size:1.7rem;color:#f5a623;">{xgb_forecast:.1f}</span> <span style="color:#8a93a6;">m3/s</span>'
        f'</div>'
    )
with f3:
    naive_err = abs(naive_forecast - actual_next_day)
    xgb_err = abs(xgb_forecast - actual_next_day)
    winner = "XGBoost" if xgb_err < naive_err else "Naive baseline"
    st.html(
        f'<div class="card">'
        f'<div class="card-title">Actual next day</div>'
        f'<span class="mono" style="font-size:1.7rem;color:#22d3ee;">{actual_next_day:.1f}</span> <span style="color:#8a93a6;">m3/s</span>'
        f'<div style="margin-top:0.5rem;color:#8a93a6;font-size:0.85rem;">'
        f'Closer on this day: <b style="color:#e8ecf3;">{winner}</b> '
        f'(baseline off by {naive_err:.1f}, XGBoost off by {xgb_err:.1f} m3/s)'
        f'</div>'
        f'</div>'
    )


# ----------------------------------------------------------------------------------------
# Flow-duration curve
# ----------------------------------------------------------------------------------------

st.html(f'<div class="section-label">Flow-duration curve{tooltip("A standard hydrology chart: it ranks every day on record from highest flow to lowest, then plots discharge against what percentage of the time that flow level is met or exceeded. Used to size hydropower systems against how reliably water is actually available.")}</div>')

sorted_flow = discharge_df["discharge_cms"].sort_values(ascending=False).reset_index(drop=True)
exceedance_pct = (sorted_flow.index + 1) / len(sorted_flow) * 100

fdc_fig = go.Figure()
fdc_fig.add_trace(go.Scatter(
    x=exceedance_pct, y=sorted_flow, mode="lines", name="Flow-duration curve",
    line=dict(color="#22d3ee", width=2.2),
))
fdc_fig.add_trace(go.Scatter(
    x=[100 - flow_percentile], y=[today_discharge], mode="markers", name=f"{selected_ts.date()}",
    marker=dict(color="#f5a623", size=13, line=dict(color="#e8ecf3", width=1.5)),
))
fdc_fig.add_hline(y=design_flow, line_dash="dot", line_color="#8a93a6",
                   annotation_text=f"Design flow (Q{int((1-DESIGN_FLOW_QUANTILE)*100)}): {design_flow:.0f} m3/s",
                   annotation_font_color="#8a93a6")
fdc_fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#e8ecf3"),
    xaxis_title="% of time flow is exceeded",
    yaxis_title="discharge (m3/s)",
    height=420,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
# Plotly's autorange for a log-scale axis combined with add_hline() has a known bug
# where the computed range can balloon to an absurd exponent (e.g. 10^105), squashing
# the real data into a flat sliver at the bottom of the chart. Setting the log-scale
# range explicitly sidesteps that broken autorange entirely.
y_min = discharge_df["discharge_cms"].min()
y_max = discharge_df["discharge_cms"].max()
fdc_fig.update_yaxes(
    type="log",
    range=[math.log10(y_min * 0.85), math.log10(y_max * 1.15)],
)

# st.container(border=True) instead of a manual div -- this section wraps a NATIVE
# Streamlit widget (st.plotly_chart), and a hand-written <div> opened here can't
# actually contain a separately-rendered widget call; Streamlit renders each st.*
# call as its own independent DOM fragment, so a raw opening/closing div pair split
# across calls like that never nests the widget inside it, it just leaves an empty,
# collapsed div sitting next to the real (unstyled) content
with st.container(border=True):
    st.plotly_chart(fdc_fig, width="stretch")
    st.caption(
        "Log y-axis, since discharge spans two orders of magnitude between drought and flood. "
        "The design flow line marks where the power conversion is calibrated to hit rated capacity."
    )


# ----------------------------------------------------------------------------------------
# Model performance
# ----------------------------------------------------------------------------------------

st.html('<div class="section-label">Model performance</div>')
single_split, cv_df = compute_performance(feature_df)

p1, p2 = st.columns([1, 1])
with p1:
    with st.container(border=True):
        st.html(f'<div class="card-title">Single 80/20 chronological split{tooltip("This split happens to place the worst flood in the record (Jan 1978) entirely in the test set -- see the cross-validation panel for whether that one split is representative.")}</div>')
        st.dataframe(single_split.style.format({"MAE (m3/s)": "{:.2f}", "RMSE (m3/s)": "{:.2f}", "R2": "{:.3f}"}), width="stretch", hide_index=True)
with p2:
    with st.container(border=True):
        st.html(f'<div class="card-title">5-fold time series cross-validation{tooltip("Expanding-window folds across the full record, so the single-split result above is checked against several different multi-year test windows.")}</div>')
        wins = int(cv_df["XGBoost wins"].sum())
        st.html(f'<span class="mono" style="font-size:1.5rem;color:#e8ecf3;">{wins} / {len(cv_df)}</span> <span style="color:#8a93a6;">folds where XGBoost beat the naive baseline on MAE</span>')
        st.dataframe(
            cv_df[["Fold", "Test window", "Baseline MAE", "XGBoost MAE", "XGBoost wins"]]
            .style.format({"Baseline MAE": "{:.2f}", "XGBoost MAE": "{:.2f}"}),
            width="stretch", hide_index=True,
        )

st.html(
    """
<div class="callout">
<b>Honest finding:</b> the naive "tomorrow = today" baseline usually matches or beats XGBoost
overall. River discharge changes gradually day to day, so simply assuming tomorrow looks like
today turns out to be a surprisingly strong forecast.
<br><br>
XGBoost's weakness shows up specifically during floods that exceed anything it saw in training.
Tree-based models can't extrapolate beyond the range of values they learned from, so once a flood
pushes discharge past that range, XGBoost gets it wrong while persistence just keeps tracking
whatever the river is actually doing.
<br><br>
This result is documented here rather than hidden, the same approach taken with the
SARIMAX-vs-XGBoost comparison on the Load Forecasting project.
</div>
"""
)


# ----------------------------------------------------------------------------------------
# Station map
# ----------------------------------------------------------------------------------------

st.html('<div class="section-label">Station location</div>')
# plotly >=6 renamed the mapbox-based trace/layout to "map" (built on MapLibre, no token
# needed, same free "carto-darkmatter" basemap style)
map_fig = go.Figure(go.Scattermap(
    lat=[STATION_LAT], lon=[STATION_LON],
    mode="markers+text",
    marker=dict(size=16, color="#f5a623"),
    text=[STATION_NAME], textposition="top right",
    textfont=dict(color="#e8ecf3", family="Space Grotesk"),
))
map_fig.update_layout(
    map=dict(style="carto-darkmatter", center=dict(lat=STATION_LAT, lon=STATION_LON), zoom=8.2),
    margin=dict(l=0, r=0, t=0, b=0),
    height=340,
    paper_bgcolor="rgba(0,0,0,0)",
)
with st.container(border=True):
    st.plotly_chart(map_fig, width="stretch")
    st.caption("Queens County, Nova Scotia -- same stretch of river as the real Mersey Hydro System's six powerhouses.")


# ----------------------------------------------------------------------------------------
# Limitations / about
# ----------------------------------------------------------------------------------------

st.html('<div class="section-label">About the power conversion</div>')
st.html(
    f"""
<div class="callout">
The real Mersey Hydro System is <b>six separate powerhouses</b>, not one turbine -- so this
dashboard simplifies it down to a single "lumped" equivalent turbine for modeling purposes.
<br><br>
That model is calibrated using a <b>design flow</b>: a discharge level chosen so that a given
percentage of days on record meet or exceed it, written as "Q" followed by that percentage.
Q{int((1-DESIGN_FLOW_QUANTILE)*100)} means the flow that's met or exceeded {(1-DESIGN_FLOW_QUANTILE)*100:.0f}% of the time -- a rare, high-flow day. This
dashboard calibrates so that Q{int((1-DESIGN_FLOW_QUANTILE)*100)} produces the real system's full {MERSEY_SYSTEM_RATED_CAPACITY_MW} MW rated output.
<br><br>
Standard small-hydro design instead uses a much more common flow, typically Q20-Q40 (met or
exceeded 20-40% of the time). Needing a rarer flow like Q{int((1-DESIGN_FLOW_QUANTILE)*100)} here to hit a realistic 42.5 MW
suggests this simplified model understates the real system's capacity -- likely because the
six actual powerhouses draw on flow and storage from reservoirs across the watershed, only some
of which passes this one downstream gauge at Milton.
</div>
"""
)


# ----------------------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------------------

st.html(
    """
<div class="footer-block">
Built by Tendekai Mugomba &nbsp;|&nbsp;
<a href="https://www.linkedin.com/in/tendekai-mugomba" target="_blank">LinkedIn</a> &nbsp;|&nbsp;
<a href="https://github.com/tmugomba" target="_blank">GitHub</a>
</div>
"""
)