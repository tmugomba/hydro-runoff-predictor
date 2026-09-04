"""
Shared Water Survey of Canada (WSC) data-fetching for the Mersey River discharge project.

Both the notebooks and the Streamlit app need the same raw daily discharge series, cleaned
the same way. Centralizing that here (instead of copy-pasting fetch code into the app)
avoids the two ever silently drifting apart.

STATION: 01ED003 -- Mersey River at Milton, Queens County, NS.
RECORD: historical only, 1954 to mid-1979 (station discontinued; see availability URL below).
Because this station has no live/current data, the app treats the historical record as a
fixed, closed dataset for analysis and backtesting -- not a live feed.

Availability check (confirm before changing date ranges):
https://wateroffice.ec.gc.ca/report/data_availability_e.html?type=historical&station=01ED003&parameter_type=Flow
"""

import io
import pandas as pd
import requests

STATION_ID = "01ED003"
STATION_NAME = "Mersey River at Milton"
STATION_LAT = 44.0546
STATION_LON = -64.7469

WSC_BULK_CSV_URL = "https://wateroffice.ec.gc.ca/services/daily_data/csv/inline"

# local fallback used when the network isn't reachable (e.g. this app's deployment
# environment has no outbound internet, or the live WSC endpoint is temporarily down) --
# same raw file the notebooks were built from
LOCAL_RAW_CSV_PATH = "data/raw/mersey_daily_discharge_raw.csv"


def fetch_wsc_discharge(
    station: str = STATION_ID,
    start_date: str = "1954-01-01",
    end_date: str = "1979-12-31",
    local_fallback_path: str = LOCAL_RAW_CSV_PATH,
) -> pd.DataFrame:
    """
    Fetch daily mean discharge (m3/s) for a WSC station, cleaned to a simple
    DatetimeIndex + discharge_cms DataFrame.

    Tries the live WSC bulk CSV endpoint first; falls back to the locally saved raw CSV
    (same file the notebooks use) if the network request fails for any reason. This keeps
    the app working even when deployed somewhere without outbound internet access.

    station : WSC station number, e.g. "01ED003".
    start_date, end_date : "YYYY-MM-DD" strings bounding the request.
    local_fallback_path : path to the locally cached raw CSV, relative to wherever the
                          calling script/app is run from.
    """
    try:
        df = _fetch_from_wsc_api(station, start_date, end_date)
    except Exception:
        # network unavailable, endpoint changed, rate-limited, etc. -- fall back rather
        # than crash the app
        df = _load_local_fallback(local_fallback_path)

    return _clean_discharge_frame(df)


def _fetch_from_wsc_api(station: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Hit the live WSC bulk CSV endpoint directly.

    Two quirks discovered while building this, both handled here:
    1. The endpoint wants the parameter as the STRING "flow" via parameters[], not the
       numeric WSC parameter code (47) that some older docs reference.
    2. The response is UTF-8 with a leading BOM character (\\ufeff) -- if that's not
       stripped, the header-detection check below (`"Date" in line`) fails on the very
       first line and the whole parse silently breaks.
    """
    params = {
        "stations[]": station,
        "parameters[]": "flow",
        "start_date": start_date,
        "end_date": end_date,
    }
    resp = requests.get(WSC_BULK_CSV_URL, params=params, timeout=30)
    resp.raise_for_status()

    text = resp.content.decode("utf-8").lstrip("\ufeff")

    # the file has a variable-length metadata preamble before the actual header row --
    # scan line-by-line for the row that looks like the real header rather than assuming
    # a fixed number of rows to skip
    lines = text.splitlines()
    header_idx = next(
        i for i, line in enumerate(lines) if "Date" in line and "Value" in line
    )
    csv_body = "\n".join(lines[header_idx:])

    return pd.read_csv(io.StringIO(csv_body))


def _load_local_fallback(path: str) -> pd.DataFrame:
    """Load the locally saved raw CSV in the same WSC export format."""
    return pd.read_csv(path)


def _clean_discharge_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize either the live-API response or the local fallback CSV (same WSC column
    layout) into a clean DataFrame: DatetimeIndex named 'date', single 'discharge_cms'
    column, reindexed to continuous daily frequency (gaps forward-filled) since the
    feature engineering's lag/rolling logic assumes no missing dates.
    """
    date_col = next(c for c in raw.columns if c.strip().lower().startswith("date"))
    value_col = next(c for c in raw.columns if c.strip().lower().startswith("value"))

    df = raw[[date_col, value_col]].rename(
        columns={date_col: "date", value_col: "discharge_cms"}
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["discharge_cms"]).sort_values("date")
    df = df.set_index("date")

    # reindex to a continuous daily calendar so downstream lag/rolling features never
    # silently skip a gap -- short gaps are forward-filled (river flow doesn't teleport),
    # anything left over (e.g. a large gap at the very start) is dropped
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range)
    df.index.name = "date"
    df["discharge_cms"] = df["discharge_cms"].ffill(limit=3)
    df = df.dropna(subset=["discharge_cms"])

    return df
