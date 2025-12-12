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

## Automated Data Updates with GitHub Actions

To keep the Feature Group continuously updated, we use **GitHub Actions** to run an **incremental ingestion pipeline** on a schedule.

### Incremental Ingestion Strategy

Instead of reloading all historical data, each scheduled run:

1. Fetches a **recent sliding window** of BTC market data from CoinGecko (default: last 5 days)
2. Recomputes all engineered features and lagged features
3. Drops the most recent **24 hours**, since the label `label_up_24h` cannot yet be computed
4. Reads the **latest timestamp already stored** in the Feature Group
5. Inserts **only new, non-overlapping rows** into the Feature Group

### GitHub Actions Integration

The incremental pipeline is triggered via **GitHub Actions**, allowing the feature store to stay synchronized with live market data without manual intervention.

Sensitive credentials (e.g., Hopsworks API key) are stored securely using **GitHub Secrets**, following best practices.

This setup demonstrates a realistic **production-style MLOps workflow**, where external data, feature engineering, and storage are fully automated.

---

## Model Training and Evaluation

### Time-Aware Train / Test Split

Model training data is loaded **exclusively from the Hopsworks Feature View** (`crypto_featureview`), ensuring consistency between feature engineering and model training.

To prevent **temporal leakage**, we perform a **manual time-series split** instead of using a random split:

1. All feature and label data are retrieved from the Feature View
2. Data is **sorted by timestamp**
3. The first **80% of samples** are used for training
4. The remaining **20% of samples** are used for testing

This guarantees that:
- training data strictly precedes test data in time
- the model never sees future information during training
- evaluation reflects a realistic forecasting scenario

```text
Train: earliest timestamp → 80% time cutoff
Test : 80% time cutoff → latest available timestamp

