# Mersey River Run-of-River Discharge & Power Dashboard

Interactive Streamlit dashboard that forecasts next-day streamflow on the Mersey River (Nova
Scotia) and converts that discharge into an estimated power output for the real **Mersey Hydro
System** — a 42.5 MW run-of-river hydroelectric asset operated by Nova Scotia Power.

Built as the hydro pillar of a renewable-energy portfolio, alongside solar, wind, and grid-scale
storage projects. All data used is publicly available (Water Survey of Canada gauge records) — no
confidential data from any prior employment.

## Problem statement

Run-of-river hydro has no reservoir to smooth out supply the way a conventional dam does — output
tracks whatever the river is doing that day. That makes two questions worth answering with data:
1. Can next-day streamflow be forecast well enough to be useful for operations planning?
2. Given a discharge reading, what does that actually mean for the plant's power output relative
   to its rated capacity?

## Data source

- **Station:** WSC `01ED003`, Mersey River at Milton, Queens County, NS
- **Record:** 1954-12-17 to 1979-05-29 (daily mean discharge, m3/s) — the station was
  discontinued after this, so the dataset is historical and fixed rather than live
- **Volume:** 8,907 daily readings, with 23 missing days (0.3%) forward-filled during cleaning
- **Range:** mean 57.2 m3/s, median 55.8 m3/s, min 1.2 m3/s, max 731 m3/s (Jan 27, 1978 — the
  worst flood in the record)
- **Source:** [Water Survey of Canada, ECCC](https://wateroffice.ec.gc.ca/) — free, no API key
  required. [Data availability check](https://wateroffice.ec.gc.ca/report/data_availability_e.html?type=historical&station=01ED003&parameter_type=Flow)

## Method

**Feature engineering** (`src/features.py`): lag features (1, 2, 3, 7, 14, 30 days), rolling
mean/std (7-day and 30-day windows, shifted to avoid leaking the current day into its own
feature), and cyclical day-of-year encoding (`sin`/`cos`) so December 31st and January 1st are
treated as adjacent rather than as opposite ends of a number line.

**Modeling** (`notebooks/03_modeling.ipynb`): an XGBoost regressor trained to predict next-day
discharge, benchmarked against the simplest possible forecast — a naive persistence baseline
("tomorrow = today"). Validated two ways:
- A single chronological 80/20 train/test split
- A 5-fold expanding-window `TimeSeriesSplit`, to check the single-split result isn't just an
  artifact of one lucky or unlucky split

**Power conversion** (`src/power.py`): the real Mersey Hydro System is **six separate
powerhouses**, fed by six reservoirs, nine dams, and two canals — not one turbine. Since we don't
have (and shouldn't fabricate) each powerhouse's individual head and efficiency, the whole cascade
is treated as **one lumped equivalent turbine**, calibrated so that a chosen design-flow
percentile maps onto the real system's 42.5 MW rated capacity, with output capped at that rating
above design flow (modeling the real-world behaviour of excess flow being spilled).

## Results — and an honest limitation, not a hidden one

**Forecasting:** the naive persistence baseline generally matches or beats XGBoost overall (MAE
8.60 m3/s vs. 10.56 m3/s on the primary split). This isn't a failed model — it's a real finding.
River discharge is highly autocorrelated day-to-day, and XGBoost's specific weak point is flood
events that exceed anything seen in training: a tree-based model can't extrapolate past the range
of values it learned from, while persistence just tracks whatever the river is actually doing. The
5-fold cross-validation confirms this pattern holds across multiple different multi-year test
windows, not just the one split that happened to contain the 1978 flood peak — the same
transparent-about-limitations approach used for the SARIMAX-vs-XGBoost comparison on the
[Load Forecasting project](../load-forecasting-ml).

**Power conversion:** hitting a realistic 40-60% capacity factor for run-of-river hydro required
calibrating against a **Q2 design flow** (exceeded only 2% of the time) — far rarer than standard
small-hydro sizing convention (typically Q20-Q40). That's a genuine signal that the single lumped
turbine simplification undercounts the real system's combined hydraulic capacity, since the actual
six powerhouses draw on flow and storage from six reservoirs spread across the watershed — not all
of which passes this one downstream gauge at Milton. This is a defensible approximation for a
portfolio-level feasibility tool, documented plainly rather than glossed over, and is **not** a
substitute for an engineering study of the real six-powerhouse system.

## Dashboard

Run locally:
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Features an animated run-of-river schematic (reservoir → penstock/canal → lumped powerhouse →
grid), speedometer-style gauges for discharge/power/capacity factor, a flow-duration curve marking
the selected day's percentile, side-by-side next-day forecasts (naive vs. XGBoost vs. actual), and
the full model performance breakdown (single-split + cross-validation) — all with hover-tooltip
explanations for every metric.

## Project structure

```
hydro-runoff-predictor/
├── data/
│   ├── raw/                  # original WSC CSV export
│   └── processed/            # engineered feature table
├── notebooks/
│   ├── 01_eda.ipynb          # station discovery, cleaning, seasonality
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb     # baseline vs. XGBoost, cross-validation, power calibration
├── src/
│   ├── data.py               # shared WSC fetch/clean logic (notebooks + app)
│   ├── features.py           # shared feature engineering (notebooks + app)
│   └── power.py               # discharge -> power conversion
├── app/
│   ├── app.py                 # Streamlit dashboard
│   ├── gauges.py               # SVG speedometer gauge generator
│   ├── schematic.py            # animated run-of-river schematic
│   └── discharge_model.pkl     # trained XGBoost model
└── requirements.txt
```

## About the author

Tendekai Mugomba — electrical engineering student with hands-on renewable energy experience
(solar PV analytics, measurement & verification, LCOE modeling, feasibility studies).

- LinkedIn: [linkedin.com/in/tendekai-mugomba](https://www.linkedin.com/in/tendekai-mugomba)
- GitHub: [github.com/tmugomba](https://github.com/tmugomba)
