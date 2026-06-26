"""Download and process daily index data per Buczynski & Chlebus (2024).

Primary source: yfinance (Yahoo Finance).
Fallback: Stooq via manual CSV download.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Yahoo Finance tickers
YAHOO_TICKERS = {
    "wig20": "WIG20.WA",   # Warsaw Stock Exchange WIG20
    "spx":   "^GSPC",       # S&P 500
    "ftse":  "^FTSE",       # FTSE 100
}

# Date range: we need data from 2004-01-01 (so 2005 period has full training window)
# through end of 2021 (last paper period is 2020 test, needs data through end of 2020)
START_DATE = "2004-01-01"
END_DATE = "2022-01-01"

def download_yahoo(name: str, ticker: str) -> pd.DataFrame:
    import yfinance as yf
    print(f"  Fetching {ticker} from Yahoo Finance...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE,
                     progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"Yahoo returned empty dataframe for {ticker}")
    # Flatten multi-index columns if present (newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    # yfinance returns 'date' or 'datetime' depending on version
    if "datetime" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"datetime": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(RAW_DIR / f"{name}.csv", index=False)
    return df

def load_local_csv(name: str) -> pd.DataFrame:
    """Load a manually-downloaded CSV from data/raw/{name}.csv."""
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No local CSV at {path}")
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["close"]) - np.log(df["close"].shift(1))
    return df.dropna(subset=["log_return"]).reset_index(drop=True)

def get_window(df: pd.DataFrame, start_date: str, n_train: int = 1000, n_test: int = 250):
    """Return (train_returns, test_returns, test_dates) starting from start_date."""
    df = df[df["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    if len(df) < n_train + n_test:
        raise ValueError(f"Not enough data after {start_date}: have {len(df)}, need {n_train + n_test}")
    train = df.iloc[:n_train]["log_return"].values
    test = df.iloc[n_train:n_train + n_test]["log_return"].values
    test_dates = df.iloc[n_train:n_train + n_test]["date"].values
    return train, test, test_dates

if __name__ == "__main__":
    for name, ticker in YAHOO_TICKERS.items():
        print(f"Downloading {name} ({ticker})...")
        try:
            df = download_yahoo(name, ticker)
        except Exception as e:
            print(f"  Yahoo failed: {e}")
            print(f"  Trying local CSV at data/raw/{name}.csv...")
            try:
                df = load_local_csv(name)
            except Exception as e2:
                print(f"  Local CSV also failed: {e2}")
                print(f"  Manual fallback steps:")
                print(f"    1. Open https://stooq.com/q/?s={ticker.replace('^','').lower()}")
                print(f"    2. Click 'Historical Data', download CSV")
                print(f"    3. Save as data/raw/{name}.csv")
                continue
        df_ret = compute_log_returns(df)
        df_ret.to_parquet(PROCESSED_DIR / f"{name}_returns.parquet")
        print(f"  {name}: {len(df_ret)} rows, "
              f"{df_ret['date'].min().date()} to {df_ret['date'].max().date()}")