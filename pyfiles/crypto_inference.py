import os
import json
import pathlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

import hopsworks
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

FG_NAME = "crypto_fg"
FG_VERSION = 1

MODEL_NAME = "crypto_xgboost_direction_model"
MODEL_VERSION = 7

TARGET_OFFSET_DAYS = int(os.getenv("TARGET_OFFSET_DAYS", "2"))

OUT_DIR = pathlib.Path("daily_inference_reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Masking / Sanitization ----------

def _get_model_feature_names(model: XGBClassifier) -> list[str]:
    """
    Get feature names the model expects.
    Works for XGBoost sklearn wrapper.
    """
    booster = model.get_booster()
    names = booster.feature_names
    if not names:
        raise ValueError("Model booster has no feature_names. Train with pandas DataFrame to store names.")
    return list(names)


def sanitize_X_for_xgb(df: pd.DataFrame, model_feature_names: list[str]) -> pd.DataFrame:
    """
    Build X strictly matching training feature set:
    - Drop non-feature columns
    - Ensure columns = model_feature_names (same names & order)
    """
    X = df.copy()

    # remove non-features if present
    drop_cols = [c for c in ["timestamp", "y_true", "hour", "price_d1"] if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    # Ensure numeric
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Align to training feature names
    missing = [c for c in model_feature_names if c not in X.columns]
    extra = [c for c in X.columns if c not in model_feature_names]
    if missing:
        raise ValueError(f"Missing required features for model: {missing}")
    if extra:
        # extra columns won't be passed to model
        X = X.drop(columns=extra)

    X = X[model_feature_names]
    return X


# ---------- Hopsworks ----------

def hopsworks_login():
    load_dotenv()
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT", "ID2223_airquality")
    if not api_key:
        raise ValueError("Missing HOPSWORKS_API_KEY")

    print("Logging into Hopsworks ...")
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    return project.get_feature_store(), project.get_model_registry()


def load_model_from_registry(mr, name: str, version: int) -> XGBClassifier:
    print(f"Loading model '{name}' version={version} from Model Registry ...")
    m = mr.get_model(name=name, version=version)

    model_dir = pathlib.Path(m.download())
    model_path = model_dir / "model.json"
    if not model_path.exists():
        candidates = list(model_dir.rglob("model.json"))
        if not candidates:
            raise FileNotFoundError(f"model.json not found under {model_dir}")
        model_path = candidates[0]

    model = XGBClassifier()
    model.load_model(str(model_path))
    print(f"Loaded model '{name}' v{version} from: {model_path}")
    return model


# ---------- Data read & hourly ----------

def _read_day_raw(fs, day_utc):
    start = datetime(day_utc.year, day_utc.month, day_utc.day, 0, 0, 0)
    end = start + timedelta(days=1)

    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
    q = fg.select_all().filter(fg.timestamp >= start).filter(fg.timestamp < end)
    df = q.read()
    if df is None or df.empty:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if getattr(df["timestamp"].dtype, "tz", None) is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)

    return df.sort_values("timestamp").reset_index(drop=True)


def to_hourly_last(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts_hour"] = df["timestamp"].dt.floor("h")
    df = df.sort_values("timestamp").drop_duplicates(subset=["ts_hour"], keep="last")
    df["timestamp"] = df["ts_hour"]
    return df.drop(columns=["ts_hour"]).sort_values("timestamp").reset_index(drop=True)


# ---------- Truth (align by hour-of-day) ----------

def build_eval_by_hour(df_d: pd.DataFrame, df_d1: pd.DataFrame) -> pd.DataFrame:
    """
    Returns df based on day D rows (features) with:
      - hour
      - price_d1 (next day price for audit)
      - y_true computed by price(D+1,h) > price(D,h)
    IMPORTANT: keep day D feature column 'price' unchanged!
    """
    if df_d.empty or df_d1.empty:
        return pd.DataFrame()

    if "price" not in df_d.columns or "price" not in df_d1.columns:
        raise ValueError("Need 'price' in both days to build truth labels.")

    d = df_d.copy()
    d1 = df_d1.copy()

    d["hour"] = d["timestamp"].dt.hour
    d1["hour"] = d1["timestamp"].dt.hour

    d = d.sort_values("timestamp").drop_duplicates(subset=["hour"], keep="last")
    d1 = d1.sort_values("timestamp").drop_duplicates(subset=["hour"], keep="last")

    # Merge by hour-of-day; keep D features
    merged = d.merge(
        d1[["hour", "price"]].rename(columns={"price": "price_d1"}),
        on="hour",
        how="inner"
    )

    # y_true uses D price (still named 'price') vs next-day price_d1
    merged["y_true"] = (merged["price_d1"] > merged["price"]).astype(int)
    merged = merged.sort_values("hour").reset_index(drop=True)
    return merged

import numpy as np
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns


def save_inference_images(
    images_dir: pathlib.Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    hours: np.ndarray,
    acc: float,
    auc: float,
    eval_date_str: str,
):
    images_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # 1) Confusion Matrix
    # =========================================================
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5.4, 4.6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        cbar=False,
        linewidths=0.6,
        linecolor="white",
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.xticks([0.5, 1.5], ["Down / 0", "Up / 1"])
    plt.yticks([0.5, 1.5], ["Down / 0", "Up / 1"], rotation=0)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title(
        f"Confusion Matrix – BTC 1h Up/Down\nEvaluation date: {eval_date_str} (UTC)",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(images_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    # =========================================================
    # 2) Hourly Prediction vs Ground Truth (unchanged)
    # =========================================================
    order = np.argsort(hours)
    h = hours[order]
    yt = y_true[order]
    yp = y_proba[order]

    plt.figure(figsize=(9.6, 4.6))
    plt.plot(h, yp, linewidth=2, color="#1f77b4", label="Predicted P(up)")
    plt.plot(h, yt, linewidth=2, color="#d62728", label="True label (0/1)")
    plt.ylim(-0.05, 1.05)
    plt.xticks(range(0, 24))
    plt.xlabel("Hour (UTC)")
    plt.ylabel("Probability / Label")
    plt.title(
        f"Hourly Prediction vs Truth – {eval_date_str} (UTC)\n"
        f"Accuracy={acc:.3f}, AUC={auc if not np.isnan(auc) else float('nan'):.3f}",
        fontsize=12
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(images_dir / "prob_vs_truth.png", dpi=200)
    plt.close()

    # =========================================================
    # 3) Hourly True vs Predicted Labels as Table
    # =========================================================
    import pandas as pd

    table_df = pd.DataFrame({
        "Hour (UTC)": h,
        "True Label": yt,
        "Predicted Label": y_pred[order],
    })

    fig, ax = plt.subplots(figsize=(6, len(table_df)*0.5 + 1))
    ax.axis("tight")
    ax.axis("off")
    the_table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1, 1.2)  # 调整表格行高
    plt.title(f"Hourly True vs Predicted Labels – {eval_date_str} (UTC)", fontsize=12)
    plt.tight_layout()
    plt.savefig(images_dir / "labels_vs_prediction.png", dpi=200)
    plt.close()



# ---------- Evaluate & Save ----------

def evaluate_and_save(model: XGBClassifier, df_eval: pd.DataFrame, target_date_str: str, missing_hours_d, missing_hours_d1):
    if df_eval.empty:
        print("No rows to evaluate.")
        return

    model_features = _get_model_feature_names(model)
    X = sanitize_X_for_xgb(df_eval, model_features)

    # Mask rows with NaN features
    good_mask = ~X.isna().any(axis=1)
    if good_mask.sum() < len(X):
        dropped = len(X) - good_mask.sum()
        print(f"[Masking] Dropped {dropped} rows with NaN features.")
        X = X.loc[good_mask].copy()
        df_eval = df_eval.loc[good_mask].copy()

    if len(X) == 0:
        print("No rows left after masking.")
        return

    y_true = df_eval["y_true"].astype(int).values
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    print("\n===== Daily Inference Evaluation (UTC) =====")
    print(f"Model             : {MODEL_NAME} v{MODEL_VERSION}")
    print(f"Target UTC date D : {target_date_str}")
    print(f"Rows evaluated    : {len(y_true)} (max 24)")
    if missing_hours_d:
        print(f"[Warn] Day D missing hours   : {missing_hours_d}")
    if missing_hours_d1:
        print(f"[Warn] Day D+1 missing hours : {missing_hours_d1}")
    print(f"Accuracy          : {acc:.4f}")
    print(f"ROC-AUC           : {auc:.4f}")
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", report)

    day_dir = OUT_DIR / target_date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    out_df = df_eval[["timestamp", "hour", "y_true"]].copy()
    out_df["y_pred"] = y_pred
    out_df["y_proba_up"] = y_proba
    out_df["price_d"] = df_eval["price"].values          # D price (feature)
    out_df["price_d1"] = df_eval["price_d1"].values      # D+1 price (audit)
    out_df.to_csv(day_dir / "predictions.csv", index=False)

    metrics = {
        "model_name": MODEL_NAME,
        "model_version": int(MODEL_VERSION),
        "target_utc_date": target_date_str,
        "rows": int(len(y_true)),
        "accuracy": float(acc),
        "roc_auc": None if np.isnan(auc) else float(auc),
        "confusion_matrix": cm.tolist(),
        "missing_hours_day_D": missing_hours_d,
        "missing_hours_day_D_plus_1": missing_hours_d1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "truth_definition": "y_true[h] = 1 if price(D+1,h) > price(D,h) else 0",
    }
    (day_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (day_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    print(f"\nSaved results to: {day_dir}")
        # -------------------
    # Save images
    # -------------------
    images_dir = OUT_DIR / "images"

    # hours for x-axis
    hours = df_eval["hour"].astype(int).values if "hour" in df_eval.columns else np.arange(len(y_true))

    save_inference_images(
        images_dir=images_dir,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        hours=hours,
        acc=acc,
        auc=auc,
        eval_date_str=target_date_str,
    )

    print(f"Saved images to: {images_dir}")


# ---------- Main ----------

def main():
    now_utc = datetime.now(timezone.utc)
    target_day = now_utc.date() - timedelta(days=TARGET_OFFSET_DAYS)  # D
    next_day = target_day + timedelta(days=1)                        # D+1

    print(f"Now UTC             : {now_utc.isoformat()}")
    print(f"Evaluating day D     : {target_day} (00:00–23:00 UTC)")
    print(f"Using next day D+1   : {next_day} (00:00–23:00 UTC) to build truth labels")

    fs, mr = hopsworks_login()
    model = load_model_from_registry(mr, name=MODEL_NAME, version=MODEL_VERSION)

    df_d_raw = _read_day_raw(fs, target_day)
    df_d1_raw = _read_day_raw(fs, next_day)

    if df_d_raw.empty:
        print(f"No data found for day D={target_day} in FG.")
        return
    if df_d1_raw.empty:
        print(f"No data found for day D+1={next_day} in FG.")
        return

    df_d = to_hourly_last(df_d_raw)
    df_d1 = to_hourly_last(df_d1_raw)

    hours_d = set(df_d["timestamp"].dt.hour.tolist())
    hours_d1 = set(df_d1["timestamp"].dt.hour.tolist())
    missing_in_d = [h for h in range(24) if h not in hours_d]
    missing_in_d1 = [h for h in range(24) if h not in hours_d1]

    df_eval = build_eval_by_hour(df_d, df_d1)

    print(f"Hourly rows available: D={len(df_d)} | D+1={len(df_d1)} | aligned_eval={len(df_eval)}")

    evaluate_and_save(
        model=model,
        df_eval=df_eval,
        target_date_str=str(target_day),
        missing_hours_d=missing_in_d,
        missing_hours_d1=missing_in_d1
    )


if __name__ == "__main__":
    main()
