"""
crypto_train_model.py

"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*is_sparse.*")
warnings.filterwarnings("ignore", message=".*backend2gui.*")

import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import hopsworks
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def load_train_test_from_fv():
    load_dotenv()
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT", "ID2223_airquality")

    if api_key is None:
        raise ValueError("没找到 HOPSWORKS_API_KEY，请在 .env 里配置你的 token。")

    print("Logging into Hopsworks ...")
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()

    # 读取 Feature View v1
    print("Getting Feature View 'crypto_featureview' v1 ...")
    fv = fs.get_feature_view(name="crypto_featureview", version=1)

    # ==================================================
    # 修正点：使用 training_data 获取 X 和 y
    # ==================================================
    print("Reading training data (X and y) from Feature View...")
    
    # training_data() 会返回 (Features, Labels) 的元组
    # description 随便写，主要为了触发它生成/读取数据
    X_fv, y_fv = fv.training_data(
        description="crypto_full_dataset"
    )

    # 1. 拼接 X 和 y，方便统一按时间排序
    df_all = pd.concat([X_fv, y_fv], axis=1)

    # 2. 按时间排序 (非常重要！否则还是会泄露)
    print("Sorting by timestamp for manual time-series split...")
    df_all = df_all.sort_values("timestamp").reset_index(drop=True)

    # 3. 提取排序后的 Label 和 Features
    y_all = df_all["label_up_24h"]
    X_all = df_all.drop(columns=["label_up_24h"])

    # 4. 手动按时间切分 (前 80% 训练，后 20% 测试)
    split_index = int(len(df_all) * 0.8)

    X_train = X_all.iloc[:split_index]
    y_train = y_all.iloc[:split_index]

    X_test = X_all.iloc[split_index:]
    y_test = y_all.iloc[split_index:]

    print(f"Time-series split complete.")
    print(f"Train: {X_train.timestamp.min()} -> {X_train.timestamp.max()} (Size: {len(X_train)})")
    print(f"Test : {X_test.timestamp.min()} -> {X_test.timestamp.max()} (Size: {len(X_test)})")

    # 5. 最后去掉 timestamp 列 (XGBoost 不需要它，且防止作为唯一ID被过拟合)
    X_train = X_train.drop(columns=["timestamp"])
    X_test = X_test.drop(columns=["timestamp"])

    return project, X_train, X_test, y_train, y_test

# ----------------------------------------------------------------------
# 2. 训练 XGBoost 模型
# ----------------------------------------------------------------------

def train_xgb_classifier(X_train, y_train, X_test, y_test):
    """
    训练一个 XGBoost 二分类模型，返回训练好的模型和一些指标。
    """
    
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    print("Training XGBoost classifier ...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",  # 更快
    )
    
    def _sanitize_X_for_xgb(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
    
        # 1) 最推荐：直接扔掉 timestamp（你已经有 hour_of_day/day_of_week/is_weekend）
        if "timestamp" in X.columns:
            X = X.drop(columns=["timestamp"])
    
        # 2) 如果还有 object 列，最好也处理掉（避免下次再踩雷）
        obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
        if obj_cols:
            print("Dropping non-numeric object columns:", obj_cols)
            X = X.drop(columns=obj_cols)

        return X


    X_train = _sanitize_X_for_xgb(X_train)
    X_test  = _sanitize_X_for_xgb(X_test)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )


    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        roc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc = float("nan")  # 万一某一类全被预测成同一类，AUC 可能报错

    print(f"Accuracy: {acc:.4f}, ROC AUC: {roc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    return model, y_pred, y_proba, acc, roc


# ----------------------------------------------------------------------
# 3. 可视化：预测 vs 真实 & feature importance & 混淆矩阵
# ----------------------------------------------------------------------


def ensure_dirs():
    """
    创建 images/ 和 models/ 目录
    """
    base_dir = pathlib.Path(".")
    images_dir = base_dir / "images_crypto"
    model_dir = base_dir / "crypto_model"

    images_dir.mkdir(exist_ok=True, parents=True)
    model_dir.mkdir(exist_ok=True, parents=True)

    return images_dir, model_dir


def plot_predictions(y_test, y_proba, images_dir: pathlib.Path):
    """
    类似老师 PM2.5 的时间序列图，这里画 test 集的预测概率 vs 真实标签。
    x 轴用样本 index。
    """
    print("Plotting predictions ...")
    idx = np.arange(len(y_test))

    plt.figure(figsize=(10, 5))
    plt.plot(idx, y_test.values, label="Actual label (0/1)", linewidth=1)
    plt.plot(idx, y_proba, label="Predicted probability (up)", linewidth=1)
    plt.xlabel("Test sample index")
    plt.ylabel("Label / Probability")
    plt.title("BTC 1h Up Movement - Actual vs Predicted Probability")
    plt.legend()
    plt.tight_layout()

    path = images_dir / "pred_vs_actual.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved prediction plot to {path}")


def plot_confusion(y_test, y_pred, images_dir: pathlib.Path):
    """
    画混淆矩阵。
    """
    print("Plotting confusion matrix ...")
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Down / 0", "Up / 1"],
        yticklabels=["Down / 0", "Up / 1"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - BTC 1h Up/Down")
    plt.tight_layout()

    path = images_dir / "confusion_matrix.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved confusion matrix to {path}")


def plot_feature_importance(model, images_dir: pathlib.Path):
    """
    利用 xgboost.plot_importance 画特征重要性。
    """
    print("Plotting feature importance ...")
    plt.figure(figsize=(8, 6))
    plot_importance(model, max_num_features=20, importance_type="gain")
    plt.tight_layout()

    path = images_dir / "feature_importance.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved feature importance plot to {path}")


# ----------------------------------------------------------------------
# 4. 保存模型 & 注册到 Model Registry
# ----------------------------------------------------------------------


def save_and_register_model(project,
                            model,
                            X_train,
                            y_train,
                            model_dir: pathlib.Path,
                            acc: float,
                            roc: float):
    """
    - 把 XGBoost 模型保存到 model_dir/model.json
    - 使用 Hopsworks Model Registry 注册一个 Python Model
    """

    print("Saving model to local directory ...")
    model_path = model_dir / "model.json"
    model.save_model(str(model_path))
    print(f"XGBoost model saved to {model_path}")

    metrics = {
        "accuracy": str(acc),
        "roc_auc": str(roc),
    }

    print("Creating model schema ...")
    
    input_schema = Schema(X_train)

    # y_train 可能是 Series 或 1列 DataFrame，统一成 1列 DataFrame
    if isinstance(y_train, pd.Series):
        y_schema_df = y_train.to_frame(name="label_up_24h")
    else:
        # DataFrame：确保只有一列，并命名为 label_up_6h
        y_schema_df = y_train.copy()
        if y_schema_df.shape[1] != 1:
            raise ValueError(f"Expected y_train to have 1 column, got {y_schema_df.shape[1]}")
        y_schema_df.columns = ["label_up_24h"]
    
    output_schema = Schema(y_schema_df)

    # input_schema = Schema(X_train)
    # output_schema = Schema(y_train.to_frame(name="label_up_6h"))
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    print(" Registering model in Hopsworks Model Registry ...")
    mr = project.get_model_registry()

    crypto_model = mr.python.create_model(
        name="crypto_xgboost_direction_model",
        description="XGBoost classifier predicting Bitcoin 1h up/down movement from engineered CoinGecko features.",
        metrics=metrics,
        model_schema=model_schema,
        input_example=X_train.iloc[:1],
    )

    # 把整个 model_dir 上传到 MR
    crypto_model.save(str(model_dir))
    print("Model registered in Hopsworks Model Registry.")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


    
def main():
    # 1) 从 Feature View 拿到 train / test (已按时间切分)
    project, X_train, X_test, y_train, y_test = load_train_test_from_fv()

    # 2) 训练模型
    model, y_pred, y_proba, acc, roc = train_xgb_classifier(X_train, y_train, X_test, y_test)

    # 3) 画图 (预测结果 & 混淆矩阵)
    images_dir, model_dir = ensure_dirs()
    plot_predictions(y_test, y_proba, images_dir)
    plot_confusion(y_test, y_pred, images_dir)
    plot_feature_importance(model, images_dir)

    # 4) 保存 & 注册模型
    save_and_register_model(project, model, X_train, y_train, model_dir, acc, roc)

    print("\nTraining pipeline finished. Starting Debugging Analysis...")

    # ==========================================
    # 🕵️‍♂️ DEBUG: 寻找泄露特征 (Safe Mode)
    # ==========================================
    
    # --- A. 特征重要性分析 ---
    importance = model.feature_importances_
    # 确保列名是列表
    feature_names = X_train.columns.tolist()
    
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features (If one is > 0.5, that's your leak):")
    print(feat_imp.head(10))
    
    # 画特征重要性图
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feat_imp.head(10))
        plt.title("Feature Importance (The Leak Detector)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not plot feature importance interactively: {e}")

    # --- B. 相关性分析 (Safe Mode) ---
    print("\nChecking Correlation with Label...")
    
    # 1. 复制 X_train 并只保留数值列 (防止字符串报错)
    debug_df = X_train.select_dtypes(include=[np.number]).copy()
    
    # 2. 安全合并 label (使用 values 避免索引不一致问题)
    # y_train 可能是 DataFrame 也可能是 Series，统一转成 numpy array
    if isinstance(y_train, pd.DataFrame):
        target_vals = y_train.iloc[:, 0].values
    else:
        target_vals = y_train.values
        
    debug_df["LABEL_TARGET"] = target_vals
    
    # 3. 计算相关性
    corr = debug_df.corr()["LABEL_TARGET"].sort_values(ascending=False)
    
    # 4. 打印结果 (排除 LABEL_TARGET 自己)
    corr = corr.drop("LABEL_TARGET", errors="ignore")
    
    print("\nTop Positive Correlations (Closer to 1.0 = Leak):")
    print(corr.head(5))
    
    print("\nTop Negative Correlations (Closer to -1.0 = Leak):")
    print(corr.tail(5))


if __name__ == "__main__":
    main()
