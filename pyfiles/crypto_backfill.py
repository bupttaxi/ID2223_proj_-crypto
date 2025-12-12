"""
crypto_backfill.py

- Fetch 90 days of Bitcoin market data (hourly granularity) from CoinGecko.
- Store result in a pandas DataFrame and preview columns + head().
"""

import os
import hopsworks
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests
from requests.exceptions import ReadTimeout, RequestException

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

def fetch_coingecko_90d_bitcoin(vs_currency: str = "usd", days: int = 90) -> pd.DataFrame:
    """
    Fetch N (default 90) days of Bitcoin market data (price, market cap, total volume) from CoinGecko.
    """
    url = f"{COINGECKO_BASE_URL}/coins/bitcoin/market_chart"
    params = {"vs_currency": vs_currency, "days": days}

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[CoinGecko] Fetching {days} days of bitcoin market data (hourly) ... "
                  f"(attempt {attempt}/{max_retries})")
            # 把超时加长一点，比如 30 秒读超时
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            break
        except ReadTimeout:
            print(f"Read timeout when calling CoinGecko on attempt {attempt}.")
            if attempt == max_retries:
                # 最后一次还超时，就把错误抛出去
                raise
        except RequestException as e:
            print(f"Error calling CoinGecko API: {e}")
            # 这类错误一般不会自动恢复，直接抛出
            raise

    # prices
    prices = pd.DataFrame(data.get("prices", []), columns=["timestamp_ms", "price"])
    # market caps
    m_caps = pd.DataFrame(data.get("market_caps", []), columns=["timestamp_ms", "market_cap"])
    # total volumes
    volumes = pd.DataFrame(data.get("total_volumes", []), columns=["timestamp_ms", "total_volume"])

    # merge by timestamp_ms
    df = prices.merge(m_caps, on="timestamp_ms", how="outer").merge(volumes, on="timestamp_ms", how="outer")

    # convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp_ms"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"[CoinGecko] Fetched {len(df)} rows of market data.")
    return df

def build_feature_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    从原始的 CoinGecko 数据（price, market_cap, total_volume, timestamp）
    构造特征 + label，并返回一个干净可训练的 DataFrame。
    """
    df = df_raw.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 确保这些列是 float 类型
    for col in ["price", "market_cap", "total_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ==========
    # 时间特征
    # ==========
    # timestamp 是 UTC 时间
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.weekday  # Monday=0, Sunday=6
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # ======================
    # 价格相关：returns / 动量
    # ======================
    # pct_change(k) = (price_t - price_{t-k}) / price_{t-k}
    df["ret_1h"] = df["price"].pct_change(1)
    df["ret_6h"] = df["price"].pct_change(6)
    df["ret_12h"] = df["price"].pct_change(12)
    df["ret_24h"] = df["price"].pct_change(24)

    # ==================
    # 波动率（基于 1h return）
    # ==================
    # 先计算 1h return，再做 rolling std
    ret_1h = df["price"].pct_change(1)
    df["volatility_6h"] = ret_1h.rolling(6).std()
    df["volatility_12h"] = ret_1h.rolling(12).std()
    df["volatility_24h"] = ret_1h.rolling(24).std()

    # ===============
    # 均线（Moving Avg）
    # ===============
    df["ma_6h"] = df["price"].rolling(6).mean()
    df["ma_12h"] = df["price"].rolling(12).mean()
    df["ma_24h"] = df["price"].rolling(24).mean()
    df["ma_diff_6_24"] = df["ma_6h"] - df["ma_24h"]

    # ========================
    # 成交量 & 市值动态（External）
    # ========================
    df["volume_change_6h"] = df["total_volume"].pct_change(6)
    df["volume_change_24h"] = df["total_volume"].pct_change(24)
    df["mcap_change_24h"] = df["market_cap"].pct_change(24)
    
    
    # ======================
    # Lagged features (自相关特征)
    # ======================
    # returns lags
    df["ret_1h_lag1"] = df["ret_1h"].shift(1)
    df["ret_1h_lag3"] = df["ret_1h"].shift(3)
    df["ret_1h_lag6"] = df["ret_1h"].shift(6)
    
    df["ret_6h_lag1"] = df["ret_6h"].shift(1)
    df["ret_6h_lag3"] = df["ret_6h"].shift(3)
    
    # volume lags (建议加 1h 的变化率，专门为 lag 做铺垫)
    df["volume_change_1h"] = df["total_volume"].pct_change(1)
    df["volume_change_1h_lag1"] = df["volume_change_1h"].shift(1)
    df["volume_change_1h_lag3"] = df["volume_change_1h"].shift(3)
    df["volume_change_1h_lag6"] = df["volume_change_1h"].shift(6)
    
    # volatility regime lags
    df["volatility_6h_lag1"] = df["volatility_6h"].shift(1)
    df["volatility_6h_lag3"] = df["volatility_6h"].shift(3)
    
    # moving-average spread lags (可选但通常有用)
    df["ma_diff_6_24_lag1"] = df["ma_diff_6_24"].shift(1)
    df["ma_diff_6_24_lag3"] = df["ma_diff_6_24"].shift(3)
    

    # ==========
    # Label 部分（12h horizon）
    # ==========
    df["future_price_24h"] = df["price"].shift(-24)
    df["label_up_24h"] = (df["future_price_24h"] > df["price"]).astype(int)
    
    # 删除未来信息列，避免泄露
    df = df.drop(columns=["future_price_24h"])

    # ==================
    # 清理 NaN & 边界行
    # ==================
    # rolling / pct_change 会在前面产生 NaN，shift(-1) 会在最后产生 NaN
    # 为了训练方便，可以简单地把含 NaN 的行全部删掉
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    return df

def write_to_crypto_fg(df_features):
    load_dotenv()
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT", "ID2223_airquality")

    if api_key is None:
        raise ValueError(" 没找到 HOPSWORKS_API_KEY，请在 .env 里配置你的 token。")

    print(" Logging into Hopsworks...")
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()

    df = df_features.copy()

    # 去重（防止 CoinGecko 偶发重复时间戳导致 primary key 冲突）
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    # Hopsworks 通常希望 timestamp 是无时区的 datetime
    if "timestamp" in df.columns and getattr(df["timestamp"].dtype, "tz", None) is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)

    print(" Creating or getting Feature Group: crypto_fg (v1) ...")
    
    fg = fs.get_or_create_feature_group(
        name="crypto_fg",
        version=1,
        primary_key=["timestamp"],
        event_time="timestamp",
        description="BTC hourly engineered features + lagged features with 1h-ahead label.",
        online_enabled=False,
    )

    print(f"Inserting {len(df)} rows into Feature Group 'crypto_fg' v1 ...")
    fg.insert(df)

    print("写入成功！df 已经存入 Hopsworks Feature Group 'crypto_fg' (v1).")


def create_fv():
    load_dotenv()
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT", "ID2223_airquality")

    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()

    # 用你刚写入的 FG v1
    crypto_fg = fs.get_feature_group(name="crypto_fg", version=1)
    query = crypto_fg.select_all()

    crypto_fv = fs.get_or_create_feature_view(
        name="crypto_featureview",
        version=1,
        labels=["label_up_24h"],
        query=query,
    )

    print("crypto_featureview v1 created!")


if __name__ == "__main__":
    # 1. 抓 CoinGecko 原始数据
    df_cg = fetch_coingecko_90d_bitcoin(vs_currency="usd")

    print("\n[RAW CoinGecko] Columns:")
    print(df_cg.columns.tolist())
    print("[RAW CoinGecko] Shape:", df_cg.shape)

    # 2. 构造特征 + label
    df_features = build_feature_dataframe(df_cg)

    print("\n[Features] Columns:")
    print(df_features.columns.tolist())
    print("[Features] Shape:", df_features.shape)

    print("\n[Features] Head:")
    print(df_features.head())
    
    write_to_crypto_fg(df_features)
    create_fv()
