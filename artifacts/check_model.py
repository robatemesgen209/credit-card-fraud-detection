
import joblib
from pathlib import Path

path = Path(__file__).resolve().parent / "fraud_model.joblib"
artifact = joblib.load(path)

print("\n========== ALL TOP-LEVEL KEYS ==========")

for key, value in artifact.items():
    print(f"\nKEY: {key}")
    print("TYPE:", type(value))
    
    if isinstance(value, dict):
        print("NESTED KEYS:", list(value.keys()))

    elif isinstance(value, (str, int, float, bool)):
        print("VALUE:", value)

print("\n========== MODEL INFORMATION ==========")

metrics = artifact.get("metrics", {})

print("Model description:", metrics.get("model"))
print("Threshold:", metrics.get("threshold"))
print("Parameters:", metrics.get("model_parameters"))

print("\n========== POSSIBLE ESTIMATORS ==========")

for key, value in artifact.items():
    if hasattr(value, "predict") or hasattr(value, "get_booster"):
        print("ESTIMATOR FOUND:", key, type(value))

print("\n========== END ==========")
