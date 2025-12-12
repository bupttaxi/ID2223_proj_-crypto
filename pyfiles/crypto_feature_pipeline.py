"""
crypto_ingest_latest.py

Incremental ingestion (recommended):
- Fetch recent window (default 5 days) of BTC hourly market data from CoinGecko
- Recompute engineered features + lagged features + label_up_24h
- Only insert "matured" rows (exclude last 24h where label_up_24h is not yet known)
- Insert into Hopsworks Feature Group: crypto_fg v1 (primary_key = timestamp)

Added (fix overlap):
- Before insert, read max(timestamp) from FG and only insert rows with timestamp > last_ts
"""

import os
import pandas as pd
import numpy as np
import requests
from requests.exceptions import ReadTimeout, RequestException
from dotenv import load_dotenv
import hopsworks

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"


# -----------------------------
# 1) Fetch recent window data
# -----------------------------
def fetch_coingecko_bitcoin(vs_currency: str = "usd", days: int = 5) -> pd.DataFrame:
    url = f"{COINGECKO_BASE_URL}/coins/bitcoin/market_chart"
    params = {"vs_currency": vs_currency, "days": days}

    max_retries = 3
    data = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[CoinGecko] Fetching {days} days of BTC market data ... (attempt {attempt}/{max_retries})")
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            break
        except ReadTimeout:
            print(f"Read timeout on attempt {attempt}.")
            if attempt == max_retries:
                raise
        except RequestException as e:
            print(f"Error calling CoinGecko API: {e}")
            raise

    prices = pd.DataFrame(data.get("prices", []), columns=["timestamp_ms", "price"])
    m_caps = pd.DataFrame(data.get("market_caps", []), columns=["timestamp_ms", "market_cap"])
    volumes = pd.DataFrame(data.get("total_volumes", []), columns=["timestamp_ms", "total_volume"])

    df = prices.merge(m_caps, on="timestamp_ms", how="outer").merge(volumes, on="timestamp_ms", how="outer")
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp_ms"]).sort_values("timestamp").reset_index(drop=True)

    print(f"[CoinGecko] Fetched rows: {len(df)}")
    return df


# -----------------------------
# 2) Feature engineering
# -----------------------------
def build_feature_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy().sort_values("timestamp").reset_index(drop=True)

    for col in ["price", "market_cap", "total_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.weekday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["ret_1h"] = df["price"].pct_change(1)
    df["ret_6h"] = df["price"].pct_change(6)
    df["ret_12h"] = df["price"].pct_change(12)
    df["ret_24h"] = df["price"].pct_change(24)

    ret_1h = df["price"].pct_change(1)
    df["volatility_6h"] = ret_1h.rolling(6).std()
    df["volatility_12h"] = ret_1h.rolling(12).std()
    df["volatility_24h"] = ret_1h.rolling(24).std()

    df["ma_6h"] = df["price"].rolling(6).mean()
    df["ma_12h"] = df["price"].rolling(12).mean()
    df["ma_24h"] = df["price"].rolling(24).mean()
    df["ma_diff_6_24"] = df["ma_6h"] - df["ma_24h"]

    df["volume_change_6h"] = df["total_volume"].pct_change(6)
    df["volume_change_24h"] = df["total_volume"].pct_change(24)
    df["mcap_change_24h"] = df["market_cap"].pct_change(24)

    df["ret_1h_lag1"] = df["ret_1h"].shift(1)
    df["ret_1h_lag3"] = df["ret_1h"].shift(3)
    df["ret_1h_lag6"] = df["ret_1h"].shift(6)

    df["ret_6h_lag1"] = df["ret_6h"].shift(1)
    df["ret_6h_lag3"] = df["ret_6h"].shift(3)

    df["volume_change_1h"] = df["total_volume"].pct_change(1)
    df["volume_change_1h_lag1"] = df["volume_change_1h"].shift(1)
    df["volume_change_1h_lag3"] = df["volume_change_1h"].shift(3)
    df["volume_change_1h_lag6"] = df["volume_change_1h"].shift(6)

    df["volatility_6h_lag1"] = df["volatility_6h"].shift(1)
    df["volatility_6h_lag3"] = df["volatility_6h"].shift(3)

    df["ma_diff_6_24_lag1"] = df["ma_diff_6_24"].shift(1)
    df["ma_diff_6_24_lag3"] = df["ma_diff_6_24"].shift(3)

    # label (24h horizon)
    df["future_price_24h"] = df["price"].shift(-24)
    df["label_up_24h"] = (df["future_price_24h"] > df["price"]).astype(int)
    df = df.drop(columns=["future_price_24h"])

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


# -----------------------------
# 3) Get FG max timestamp (NEW)
# -----------------------------
def get_fg_max_timestamp(fg) -> pd.Timestamp | None:
    """
    Read max timestamp in FG to filter overlaps.
    If FG is empty, return None.
    """
    try:
        # 只读 timestamp 一列，尽量轻量
        df_ts = fg.select(["timestamp"]).read()
        if df_ts is None or df_ts.empty:
            return None
        last_ts = pd.to_datetime(df_ts["timestamp"]).max()
        # Hopsworks 读出来通常是 tz-naive；统一成 tz-naive
        if getattr(last_ts, "tzinfo", None) is not None:
            last_ts = last_ts.tz_convert(None)
        return last_ts
    except Exception as e:
        print(f"Failed to read max(timestamp) from FG, will insert as-is. Error: {e}")
        return None


# -----------------------------
# 4) Hopsworks write (FILTER OVERLAP HERE)
# -----------------------------
def write_to_crypto_fg(df_features: pd.DataFrame):
    load_dotenv()
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT", "ID2223_airquality")

    if not api_key:
        raise ValueError(" Missing HOPSWORKS_API_KEY in .env")

    print("Logging into Hopsworks...")
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()

    fg = fs.get_feature_group(name="crypto_fg", version=1)

    df = df_features.copy()

    # 统一 timestamp 为 tz-naive（和 FG 读出来的 last_ts 对齐）
    if getattr(df["timestamp"].dtype, "tz", None) is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)

    # 去重（保险）
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    # 关键：去掉 overlap，只保留 FG 里没有的新行
    last_ts = get_fg_max_timestamp(fg)
    if last_ts is not None:
        before = len(df)
        df = df[df["timestamp"] > last_ts].copy()
        after = len(df)
        print(f"Overlap filter: last_ts_in_fg={last_ts} | keep {after}/{before} rows")

    if df.empty:
        print(" No new rows to insert (everything overlaps).")
        return

    print(f" Inserting {len(df)} NEW matured rows into Feature Group 'crypto_fg' v1 ...")
    fg.insert(df)
    print(" Insert done.")


# -----------------------------
# 5) Main incremental pipeline
# -----------------------------
def main(vs_currency: str = "usd", days_window: int = 5):
    df_raw = fetch_coingecko_bitcoin(vs_currency=vs_currency, days=days_window)

    df_feat = build_feature_dataframe(df_raw)
    if df_feat.empty:
        print(" No rows after feature engineering (maybe window too small).")
        return

    # keep matured rows (exclude last 24h relative to max ts in engineered df)
    max_ts = df_feat["timestamp"].max()
    cutoff = max_ts - pd.Timedelta(hours=24)
    df_mature = df_feat[df_feat["timestamp"] <= cutoff].copy()

    if df_mature.empty:
        print(" No matured rows to write (need more history window). Try days_window=6 or 7.")
        return

    print(f" Engineered rows: {len(df_feat)} | Matured rows (pre-filter): {len(df_mature)}")
    print(f" Matured time range: {df_mature['timestamp'].min()} -> {df_mature['timestamp'].max()}")

    write_to_crypto_fg(df_mature)


if __name__ == "__main__":
    main(vs_currency="usd", days_window=5)
