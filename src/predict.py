"""Strict, reusable inference helpers for the Module 6 fraud prototype."""
from pathlib import Path
import math
import joblib
import pandas as pd

DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "fraud_model.joblib"

class InputValidationError(ValueError):
    pass

def load_artifact(path=DEFAULT_ARTIFACT):
    if not Path(path).exists():
        raise FileNotFoundError(f"Model artifact not found: {path}. Run `python -m src.train`.")
    return joblib.load(path)

def validate_transaction(payload, feature_names):
    if not isinstance(payload, dict):
        raise InputValidationError("Transaction must be a JSON object.")
    supplied = set(payload)
    expected = set(feature_names)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing:
        raise InputValidationError(f"Missing feature(s): {', '.join(missing)}")
    if extra:
        raise InputValidationError(f"Unexpected feature(s): {', '.join(extra)}")
    values = {}
    for name in feature_names:
        value = payload[name]
        if isinstance(value, bool):
            raise InputValidationError(f"{name} must be a finite number.")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise InputValidationError(f"{name} must be a finite number.")
        if not math.isfinite(number):
            raise InputValidationError(f"{name} must be a finite number.")
        values[name] = number
    return pd.DataFrame([values], columns=feature_names)

def score_transaction(payload, artifact=None):
    artifact = artifact or load_artifact()
    row = validate_transaction(payload, artifact["feature_names"])
    probability = float(artifact["pipeline"].predict_proba(row)[0, 1])
    threshold = float(artifact["threshold"])
    return {
        "model_version": artifact["model_version"],
        "fraud_probability": probability,
        "threshold": threshold,
        "decision": "review" if probability >= threshold else "no_review",
        "disclaimer": "Decision-support signal only; not proof of fraud and not an automatic card block."
    }
