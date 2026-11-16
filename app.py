import streamlit as st
import pandas as pd
from src.predict import load_artifact, score_transaction, InputValidationError

st.set_page_config(page_title="Credit Card Fraud Review", page_icon="💳", layout="wide")
st.title("Credit Card Fraud Review")
st.caption("Module 6 prototype | Cost-sensitive XGBoost | Human-in-the-loop decision support")
artifact = load_artifact()
st.warning("A model score is not proof of fraud. This prototype does not block cards or replace analyst review.")
with st.expander("Model and evaluation information"):
    st.write(f"Model version: `{artifact['model_version']}`")
    st.write(f"Decision threshold: `{artifact['threshold']:.6f}`")
    st.json(artifact["metrics"])
st.subheader("Score one transaction")
st.write("Enter the 30 numeric features from the benchmark dataset.")
with st.form("transaction_form"):
    cols = st.columns(3)
    values = {}
    for i, name in enumerate(artifact["feature_names"]):
        with cols[i % 3]:
            values[name] = st.number_input(name, value=0.0, format="%.8f")
    submitted = st.form_submit_button("Score transaction", type="primary")
if submitted:
    try:
        result = score_transaction(values, artifact)
        c1, c2 = st.columns(2)
        c1.metric("Fraud risk score", f"{result['fraud_probability']:.2%}")
        c2.metric("Decision", result["decision"].replace("_", " ").upper())
        st.write(f"Threshold: {result['threshold']:.6f}")
        st.info(result["disclaimer"])
    except InputValidationError as e:
        st.error(str(e))
st.divider()
st.subheader("Batch scoring")
upload = st.file_uploader("Upload a CSV containing the 30 feature columns (Class is optional and ignored)", type=["csv"])
if upload:
    batch = pd.read_csv(upload)
    # The benchmark CSV includes the target label; it is not a model input.
    # Drop it before validating the 30 feature columns.
    if "Class" in batch.columns:
        batch = batch.drop(columns=["Class"])
    missing = sorted(set(artifact["feature_names"]) - set(batch.columns))
    extra = sorted(set(batch.columns) - set(artifact["feature_names"]))
    if missing or extra:
        st.error(f"CSV schema mismatch. Missing: {missing}; unexpected: {extra}")
    else:
        probs = artifact["pipeline"].predict_proba(batch[artifact["feature_names"]])[:, 1]
        out = batch.copy()
        out["fraud_probability"] = probs
        out["decision"] = ["review" if x >= artifact["threshold"] else "no_review" for x in probs]
        st.dataframe(out.head(100), use_container_width=True)
        st.download_button("Download scored CSV", out.to_csv(index=False), "scored_transactions.csv", "text/csv")
