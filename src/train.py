"""Train the Module 5-aligned cost-sensitive XGBoost model."""
from pathlib import Path
import json, time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (precision_recall_curve, average_precision_score, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "creditcard.csv"

def main():
    raw = pd.read_csv(DATA)
    required = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    if set(raw.columns) != required:
        raise ValueError("Dataset columns do not match the expected 30 features plus Class.")
    if raw.isna().any().any():
        print("Missing values found; median imputation will handle them.")
    clean = raw.drop_duplicates().reset_index(drop=True)
    X, y = clean.drop(columns="Class"), clean["Class"].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    weight = float((y_train == 0).sum() / (y_train == 1).sum())
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("model", XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=weight,
            tree_method="hist", eval_metric="logloss", random_state=42, n_jobs=-1))
    ])
    start = time.time()
    model.fit(X_train, y_train)
    training_seconds = time.time() - start
    val_prob = model.predict_proba(X_val)[:, 1]
    p, r, t = precision_recall_curve(y_val, val_prob)
    eligible = [i for i in range(len(t)) if r[i] >= 0.80]
    if eligible:
        best = max(eligible, key=lambda i: p[i])
        threshold, val_p, val_r = float(t[best]), float(p[best]), float(r[best])
    else:
        threshold, val_p, val_r = 0.5, None, None
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    results = {
        "raw_rows": len(raw), "duplicate_rows_removed": int(raw.duplicated().sum()),
        "clean_rows": len(clean), "clean_fraud_rows": int(y.sum()),
        "missing_values": int(raw.isna().sum().sum()),
        "split_rows": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "split_fraud": {"train": int(y_train.sum()), "validation": int(y_val.sum()), "test": int(y_test.sum())},
        "threshold": threshold, "validation_precision": val_p, "validation_recall": val_r,
        "average_precision": float(average_precision_score(y_test, prob)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": cm.tolist(), "training_seconds": training_seconds
    }
    artifact = {"pipeline": model, "feature_names": list(X.columns), "threshold": threshold,
        "model_version": "xgboost-cost-sensitive-module6-v1",
        "label_mapping": {"0": "legitimate", "1": "fraud"}, "metrics": results}
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(artifact, ROOT / "artifacts" / "fraud_model.joblib", compress=3)
    (ROOT / "reports" / "evaluation.json").write_text(json.dumps(results, indent=2))
    (ROOT / "reports" / "classification_report.txt").write_text(
        classification_report(y_test, pred, target_names=["legitimate", "fraud"], zero_division=0))
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
