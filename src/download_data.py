"""Download daily NSE price, volume and deliverable data for the 30 study stocks.

Uses the `nselib` package (NSE's "Security-wise Price volume & Deliverable
position" report, the same report used in the original 2021 project).

Usage:
    pip install nselib
    python src/download_data.py

Output:
    data/raw/nse_prices.csv   one row per stock per trading day
    data/raw/download_log.json  which symbols worked / failed

If NSE blocks the download (it sometimes refuses cloud servers such as
Colab), download the CSVs manually from nseindia.com > Market Data >
Historical Reports > Security-wise Price volume & Deliverable position data,
put them in data/raw/manual/, and data.load_prices() will read them instead.
"""

import json
import time
from pathlib import Path

import pandas as pd

FROM_DATE = "01-03-2020"  # a little before training starts, for lagged features
TO_DATE = "30-04-2021"

# The 30 stocks from the original project. Some companies have changed their
# NSE symbol since 2021, so later names are listed as fallbacks.
STOCKS = {
    "Tata Motors": ["TATAMOTORS", "TMPV"],
    "Reliance": ["RELIANCE"],
    "Bandhan Bank": ["BANDHANBNK"],
    "GAIL": ["GAIL"],
    "PVR": ["PVRINOX", "PVR"],
    "IRCTC": ["IRCTC"],
    "State Bank of India": ["SBIN"],
    "IDFC First Bank": ["IDFCFIRSTB"],
    "Federal Bank": ["FEDERALBNK"],
    "NRB Bearings": ["NRBBEARING"],
    "PNC Infratech": ["PNCINFRA"],
    "Wipro": ["WIPRO"],
    "ITC": ["ITC"],
    "DMart": ["DMART"],
    "Coal India": ["COALINDIA"],
    "HDFC Life": ["HDFCLIFE"],
    "SBI Life": ["SBILIFE"],
    "Bharti Airtel": ["BHARTIARTL"],
    "Vodafone Idea": ["IDEA"],
    "Tata Steel": ["TATASTEEL"],
    "JSW Steel": ["JSWSTEEL"],
    "DLF": ["DLF"],
    "Oberoi Realty": ["OBEROIRLTY"],
    "Oil India": ["OIL"],
    "Apollo Hospitals": ["APOLLOHOSP"],
    "Mrs Bectors Food": ["BECTORFOOD"],
    "Jet Airways": ["JETAIRWAYS"],
    "SpiceJet": ["SPICEJET"],
    "Jubilant FoodWorks": ["JUBLFOOD"],
    "Kotak Mahindra Bank": ["KOTAKBANK"],
}


def main() -> None:
    from nselib import capital_market

    out_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, log = [], {}
    for company, symbols in STOCKS.items():
        for sym in symbols:
            try:
                df = capital_market.price_volume_and_deliverable_position_data(
                    symbol=sym, from_date=FROM_DATE, to_date=TO_DATE)
                if df is None or df.empty:
                    raise ValueError("no rows returned")
                df["Company"] = company
                frames.append(df)
                log[company] = {"symbol": sym, "rows": len(df)}
                print(f"  ok   {company:<22} {sym:<12} {len(df)} rows")
                break
            except Exception as exc:
                print(f"  fail {company:<22} {sym:<12} {str(exc)[:80]}")
            finally:
                time.sleep(1)  # be polite to the NSE server
        else:
            log[company] = {"symbol": None, "rows": 0}

    if not frames:
        raise RuntimeError("Nothing downloaded. NSE may be blocking this computer; see the docstring.")
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(out_dir / "nse_prices.csv", index=False)
    (out_dir / "download_log.json").write_text(json.dumps(log, indent=2))
    ok = sum(1 for v in log.values() if v["symbol"])
    print(f"\nDownloaded {ok}/{len(STOCKS)} stocks, {len(raw)} rows -> {out_dir / 'nse_prices.csv'}")


if __name__ == "__main__":
    main()
