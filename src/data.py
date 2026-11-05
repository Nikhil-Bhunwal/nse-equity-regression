"""Load and clean the NSE daily data.

Accepts either the combined file written by download_data.py or CSVs
downloaded by hand from the NSE website (data/raw/manual/*.csv). Column names
differ slightly between the two, so they are normalised to short names.
"""

import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

TRAIN_START, TRAIN_END = "2020-03-17", "2021-03-16"  # as in the original project
TEST_START, TEST_END = "2021-03-17", "2021-04-18"

# normalised NSE column name -> short name
COLUMN_MAP = {
    "symbol": "symbol", "series": "series", "date": "date", "company": "company",
    "prevclose": "prev_close", "openprice": "open", "highprice": "high",
    "lowprice": "low", "lastprice": "last", "closeprice": "close",
    "averageprice": "vwap", "totaltradedquantity": "volume",
    "turnoverinrs": "turnover", "turnover": "turnover", "turnoverrs": "turnover",
    "nooftrades": "trades", "deliverableqty": "deliverable_qty",
    "dlyqttotradedqty": "deliverable_pct",
}

NUMERIC = ["prev_close", "open", "high", "low", "last", "close", "vwap", "volume",
           "turnover", "trades", "deliverable_qty", "deliverable_pct"]

# The 11 candidate explanatory variables from the original report (close is the target)
CANDIDATES = ["prev_close", "open", "high", "low", "last", "vwap", "volume",
              "turnover", "trades", "deliverable_qty", "deliverable_pct"]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: COLUMN_MAP.get(re.sub(r"[^a-z]", "", c.lower()), c))
    for col in NUMERIC:
        if col in df:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip(),
                                    errors="coerce")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")
    if "series" in df:
        df["series"] = df["series"].astype(str).str.strip()
    if "symbol" in df:
        df["symbol"] = df["symbol"].astype(str).str.strip()
    return df


def load_prices(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One row per stock per trading day, sorted by stock then date (oldest first)."""
    combined = raw_dir / "nse_prices.csv"
    if combined.exists():
        df = _normalise(pd.read_csv(combined))
    else:
        files = sorted((raw_dir / "manual").glob("*.csv"))
        if not files:
            raise FileNotFoundError("No data: run src/download_data.py or add CSVs to data/raw/manual/")
        df = pd.concat([_normalise(pd.read_csv(f)) for f in files], ignore_index=True)
    if "company" not in df:
        df["company"] = df["symbol"]

    # keep the normal equity series; a few stocks trade in BE (trade-to-trade) for a while
    if "series" in df:
        df = df[df["series"].isin(["EQ", "BE"])]
    df = (df.dropna(subset=["close", "prev_close", "low", "open", "trades"])
            .drop_duplicates(subset=["symbol", "date"])
            .sort_values(["symbol", "date"]))

    # yesterday's values, used by the look-ahead-free model
    g = df.groupby("symbol")
    df["low_lag1"] = g["low"].shift(1)
    df["trades_lag1"] = g["trades"].shift(1)
    return df.reset_index(drop=True)


def split(df: pd.DataFrame):
    train = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)]
    test = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)]
    return train, test
