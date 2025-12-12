# Crypto Price Direction Prediction (BTC)

This project builds an end-to-end **machine learning pipeline** for predicting the **24-hour price direction of Bitcoin (BTC)** using **dynamic external market data**.  
It demonstrates data ingestion, feature engineering, time-aware labeling, incremental updates, and MLOps practices using **Hopsworks** and **GitHub Actions**.

---

## Data Source: Dynamic External Features (CoinGecko)

This project uses **dynamic external data** from a public API as the primary data source.

### External Data Provider
- **CoinGecko API**
- Endpoint: `/coins/bitcoin/market_chart`
- Granularity: **hourly**
- Currency: USD

### Raw Features
The following raw features are fetched from CoinGecko:

- `price`
- `market_cap`
- `total_volume`
- `timestamp` (UTC)

These are considered **external features** because they originate from an external, continuously updated data source rather than a static dataset.

---

## Feature Engineering

From the raw CoinGecko signals, we construct a rich feature set capturing temporal patterns, momentum, volatility, and market dynamics. The features includes: Time-Based Features (`hour_of_day`, `day_of_week`, `is_weekend`), Returns & Momentum(`ret_1h`, `ret_6h`, `ret_12h`, `ret_24h`), Rolling Volatility(`volatility_6h`,`volatility_24h`),  Moving Averages(`ma_6h`,`ma_12h`,`ma_24h`,`ma_diff_6_24`), External Market Dynamics(`volume_change_6h`,`volume_change_24h`,`mcap_change_24h`), Lagged Features to capture short-term temporal dependencies(`ret_1h_lag1`, `ret_1h_lag3`, `ret_1h_lag6`, `volume_change_1h_lag1`, `volume_change_1h_lag3`, `volume_change_1h_lag6`, `volatility_6h_lag1`, `volatility_6h_lag3`, `ma_diff_6_24_lag1`, `ma_diff_6_24_lag3`)

## Label Definition (24h Horizon)

The prediction target is a **binary classification label**:

```text
label_up_24h = 1  if price(t + 24h) > price(t)
label_up_24h = 0  otherwise
```
---

## Feature Store Setup (Hopsworks)

To ensure reproducibility, consistency between training and inference, and proper MLOps practices, this project uses the **Hopsworks Feature Store** to manage engineered features and labels.

### Feature Group (FG)

We create a **Feature Group** in Hopsworks to store all engineered features and labels derived from CoinGecko data.

- **Name**: `crypto_fg`
- **Primary key**: `timestamp`
- **Event time**: `timestamp`
- **Data**:
  - Raw external features (price, market cap, volume)
  - Engineered features (returns, volatility, moving averages)
  - Lagged features
  - Label (`label_up_24h`)

The Feature Group is initially populated using a **backfill pipeline** that loads approximately 90 days of historical hourly data.

### Feature View (FV)

On top of the Feature Group, we create a **Feature View** to define the exact dataset used for model training.

- **Name**: `crypto_featureview`
- **Label**: `label_up_24h`
- **Query**: all features from `crypto_fg`

The Feature View provides a clean abstraction for:
- training / test data retrieval
- time-series splits
- consistent feature-label alignment

All training pipelines read data **only through the Feature View**, not directly from raw tables.

---

