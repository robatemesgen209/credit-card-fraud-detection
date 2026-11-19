from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
APP_TITLE = "Sentinel | Card Fraud Intelligence"
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "artifacts" / "fraud_model.joblib"
SCHEMA_PATH = BASE_DIR / "reports" / "feature_schema.json"

DEFAULT_THRESHOLD = 0.535887

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: #0b1120; color: #e5e7eb; }
      [data-testid="stSidebar"] { background: #101a2d; border-right: 1px solid #24324a; }
      [data-testid="stHeader"] { background: rgba(11,17,32,.85); }
      .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
      .hero {
        padding: 1.5rem 1.7rem; border-radius: 18px;
        background: linear-gradient(120deg, #172b4d 0%, #12213a 55%, #172554 100%);
        border: 1px solid #294466; margin-bottom: 1.2rem;
      }
      .hero h1 { margin: 0; color: #f8fafc; font-size: 2rem; font-weight: 700; }
      .hero p { margin: .45rem 0 0 0; color: #b6c7df; }
      .eyebrow { color: #60a5fa; font-size: .75rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
      div[data-testid="stMetric"] {
        background: #121e32; border: 1px solid #263750;
        padding: 1rem 1.1rem; border-radius: 14px;
      }
      div[data-testid="stMetricLabel"] { color: #9fb0c7; }
      div[data-testid="stMetricValue"] { color: #f8fafc; }
      .panel {
        background: #121e32; border: 1px solid #263750;
        border-radius: 14px; padding: 1rem 1.2rem;
      }
      .muted { color: #9fb0c7; }
      .status-review {
        background: rgba(239,68,68,.12); color: #fca5a5;
        border: 1px solid rgba(239,68,68,.4); padding: .8rem 1rem;
        border-radius: 12px; font-weight: 700;
      }
      .status-clear {
        background: rgba(34,197,94,.10); color: #86efac;
        border: 1px solid rgba(34,197,94,.35); padding: .8rem 1rem;
        border-radius: 12px; font-weight: 700;
      }
      .small-note { color: #94a3b8; font-size: .82rem; }
      div.stButton > button[kind="primary"] {
        background: #2563eb; border: 0; border-radius: 10px;
        font-weight: 700; min-height: 2.8rem;
      }
      div.stButton > button[kind="primary"]:hover { background: #1d4ed8; }
      hr { border-color: #263750; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Model and configuration helpers
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading fraud detection model...")
def load_model(model_path):
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at: {path}\n"
            "Please check your artifacts folder."
        )

    artifact = joblib.load(path)

    # The artifact is saved as a dict; unwrap the pipeline from it.
    if isinstance(artifact, dict):
        pipeline = artifact.get("pipeline")
        if pipeline is None:
            raise ValueError(
                "Model artifact is a dict but contains no 'pipeline' key. "
                f"Available keys: {list(artifact.keys())}"
            )
        return pipeline

    return artifact


def load_threshold() -> float:
    """Read threshold from the model artifact first; fall back to report files, then the default."""
    artifact_path = MODEL_PATH
    if artifact_path.exists():
        try:
            raw = joblib.load(artifact_path)
            if isinstance(raw, dict):
                value = raw.get("threshold")
                if value is not None:
                    value = float(value)
                    if 0.0 <= value <= 1.0:
                        return value
        except Exception:
            pass

    possible_files = [
        Path("reports/evaluation.json"),
        Path("reports/threshold.json"),
    ]
    for path in possible_files:
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                for key in ("threshold", "best_threshold", "decision_threshold"):
                    value = payload.get(key)
                    if value is not None:
                        value = float(value)
                        if 0.0 <= value <= 1.0:
                            return value
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
    return DEFAULT_THRESHOLD


def get_feature_order() -> list[str]:
    """Prefer the project's feature schema if it contains a usable feature list."""
    if SCHEMA_PATH.exists():
        try:
            schema: Any = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
            if isinstance(schema, list):
                candidate = schema
            elif isinstance(schema, dict):
                candidate = (
                    schema.get("features")
                    or schema.get("feature_names")
                    or schema.get("columns")
                )
            else:
                candidate = None

            if (
                isinstance(candidate, list)
                and len(candidate) == len(FEATURES)
                and set(candidate) == set(FEATURES)
            ):
                return candidate
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return FEATURES


def probability_scores(model, frame: pd.DataFrame) -> np.ndarray:
    """Return the positive-class probability for each transaction."""
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "The loaded model does not expose predict_proba(). "
            "Use the trained probability-enabled classifier or adapt this function "
            "to your model's prediction interface."
        )

    probabilities = np.asarray(model.predict_proba(frame))
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Model returned an unexpected probability output shape.")
    return probabilities[:, 1].astype(float)


def validate_and_prepare(frame: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    """Validate user data and return features in the exact training order."""
    clean = frame.copy()
    clean.columns = [str(column).strip() for column in clean.columns]

    # The training target must never be used as an input feature.
    if TARGET_COLUMN in clean.columns:
        clean = clean.drop(columns=[TARGET_COLUMN])

    missing = [feature for feature in feature_order if feature not in clean.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    clean = clean[feature_order].copy()
    for column in feature_order:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    if clean.isna().any().any():
        bad_columns = clean.columns[clean.isna().any()].tolist()
        raise ValueError(
            "Some required values are empty or non-numeric in: "
            + ", ".join(bad_columns)
        )

    if not np.isfinite(clean.to_numpy(dtype=float)).all():
        raise ValueError("Input contains infinite or invalid numeric values.")

    return clean


def analyze_transactions(model, frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    scores = probability_scores(model, frame)
    results = frame.copy()
    results["Fraud Probability"] = scores
    results["Risk Score (%)"] = np.round(scores * 100, 2)
    results["Decision"] = np.where(scores >= threshold, "REVIEW", "NO REVIEW")
    results["Risk Level"] = np.select(
        [scores >= 0.80, scores >= threshold],
        ["High", "Elevated"],
        default="Low",
    )
    return results


def format_percent(value: float) -> str:
    return f"{value:.2%}"


# ---------------------------------------------------------------------
# Load model once
# ---------------------------------------------------------------------
threshold = load_threshold()
feature_order = get_feature_order()

try:
    model = load_model(str(MODEL_PATH))
    model_error = None
except Exception as exc:
    model = None
    model_error = str(exc)

# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ Sentinel")
    st.caption("CARD FRAUD INTELLIGENCE")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Single Transaction", "Batch Analysis", "About the Model"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**Model status**")
    if model is not None:
        st.success("Model loaded")
    else:
        st.error("Model unavailable")
    st.caption(f"Decision threshold: {threshold:.6f}")
    st.markdown("---")
    st.caption("Secure • Explainable • Data-driven")
    st.caption("For decision support only.")

# ---------------------------------------------------------------------
# Shared header
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Fraud prevention workspace</div>
      <h1>Card Fraud Intelligence</h1>
      <p>Detect suspicious card transactions, assess risk, and review results in one workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model_error:
    st.error(
        "The model could not be loaded. Check that "
        "`artifacts/fraud_model.joblib` exists and that dependencies match "
        "the environment used to train it."
    )
    st.code(model_error)
    st.stop()

# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------
if page == "Overview":
    st.subheader("Overview")
    st.markdown(
        '<p class="muted">Your fraud monitoring workspace. Upload transactions in Batch Analysis '
        "or assess one transaction at a time.</p>",
        unsafe_allow_html=True,
    )

    if "last_results" in st.session_state and not st.session_state.last_results.empty:
        results = st.session_state.last_results
        total = len(results)
        flagged = int((results["Decision"] == "REVIEW").sum())
        cleared = total - flagged
        fraud_rate = flagged / total if total else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions analyzed", f"{total:,}")
        c2.metric("Flagged for review", f"{flagged:,}")
        c3.metric("Not flagged", f"{cleared:,}")
        c4.metric("Review rate", format_percent(fraud_rate))

        st.markdown("### Latest analysis")
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("**Decision distribution**")
            distribution = (
                results["Decision"]
                .value_counts()
                .reindex(["NO REVIEW", "REVIEW"], fill_value=0)
                .rename_axis("Decision")
                .reset_index(name="Transactions")
            )
            st.bar_chart(distribution, x="Decision", y="Transactions", height=260)
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("**Risk score distribution**")
            st.histogram if False else None
            st.bar_chart(
                results["Risk Score (%)"].round(-1).value_counts().sort_index(),
                height=260,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Recent transactions")
        display_columns = [
            col for col in
            ["Time", "Amount", "Risk Score (%)", "Risk Level", "Decision"]
            if col in results.columns
        ]
        st.dataframe(
            results[display_columns].tail(10),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "No transactions have been analyzed in this session yet. "
            "Open **Single Transaction** to test one transaction or "
            "**Batch Analysis** to upload a CSV."
        )
        st.markdown("### How it works")
        a, b, c = st.columns(3)
        with a:
            st.markdown("**1 · Provide data**")
            st.write("Enter a transaction or upload a CSV containing the 30 model features.")
        with b:
            st.markdown("**2 · Score risk**")
            st.write("The trained model calculates a probability score for each transaction.")
        with c:
            st.markdown("**3 · Review alerts**")
            st.write("Transactions at or above the configured threshold are marked REVIEW.")

# ---------------------------------------------------------------------
# Single transaction
# ---------------------------------------------------------------------
elif page == "Single Transaction":
    st.subheader("Single transaction assessment")
    st.write("Enter the transaction's 30 numeric model features below.")
    st.info(
        "V1–V28 are anonymized PCA features from the original dataset. "
        "They are not ordinary card details such as card number, name, or CVV. "
        "Use values prepared with the same preprocessing as the training data."
    )

    with st.form("single_transaction_form"):
        st.markdown("#### Transaction information")
        col1, col2 = st.columns(2)
        with col1:
            time_value = st.number_input("Time", min_value=0.0, value=0.0, step=1.0)
        with col2:
            amount_value = st.number_input("Amount", min_value=0.0, value=100.0, step=1.0)

        with st.expander("Advanced model features · V1 to V28", expanded=True):
            feature_values = {}
            columns = st.columns(4)
            for index in range(1, 29):
                with columns[(index - 1) % 4]:
                    feature_values[f"V{index}"] = st.number_input(
                        f"V{index}",
                        value=0.0,
                        step=0.01,
                        format="%.6f",
                        key=f"single_V{index}",
                    )

        submitted = st.form_submit_button(
            "Analyze transaction",
            type="primary",
        )

    if submitted:
        row = {"Time": time_value, **feature_values, "Amount": amount_value}
        input_frame = pd.DataFrame([row])[feature_order]
        try:
            score = float(probability_scores(model, input_frame)[0])
            decision = "REVIEW" if score >= threshold else "NO REVIEW"
            risk_level = "High" if score >= 0.80 else ("Elevated" if score >= threshold else "Low")

            st.session_state.last_single = {
                "score": score,
                "decision": decision,
                "risk_level": risk_level,
                "row": row,
            }

            st.markdown("### Assessment result")
            m1, m2, m3 = st.columns(3)
            m1.metric("Fraud probability", format_percent(score))
            m2.metric("Risk level", risk_level)
            m3.metric("Decision", decision)

            if decision == "REVIEW":
                st.markdown(
                    '<div class="status-review">⚠️ REVIEW REQUIRED — This transaction reached '
                    "or exceeded the configured decision threshold.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="status-clear">✓ NO REVIEW — This transaction is below the '
                    "configured decision threshold.</div>",
                    unsafe_allow_html=True,
                )

            st.progress(min(max(score, 0.0), 1.0))
            st.caption(
                f"Model score: {score:.6f} · Decision threshold: {threshold:.6f}"
            )
            st.warning(
                "This is a model-assisted risk assessment, not proof that a transaction "
                "is fraudulent or legitimate. Follow your organization's review process."
            )
        except Exception as exc:
            st.error(f"Could not analyze this transaction: {exc}")

# ---------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------
elif page == "Batch Analysis":
    st.subheader("Batch transaction analysis")
    st.write("Upload a CSV file containing the same 30 features used during model training.")

    with st.expander("Required CSV format", expanded=False):
        st.write("Required columns:")
        st.code(", ".join(feature_order))
        st.caption(
            "An optional `Class` column is ignored. Do not include customer names, "
            "card numbers, CVVs, or other unnecessary personal data."
        )
        example = pd.DataFrame([{feature: 0.0 for feature in feature_order}])
        st.download_button(
            "Download CSV template",
            data=example.to_csv(index=False).encode("utf-8"),
            file_name="fraud_detection_template.csv",
            mime="text/csv",
        )

    uploaded_file = st.file_uploader("Choose transaction CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            raw = pd.read_csv(uploaded_file)
            st.markdown("#### File preview")
            st.caption(f"{len(raw):,} rows · {len(raw.columns):,} columns")
            st.dataframe(raw.head(8), width="stretch", hide_index=True)

            if st.button("Run fraud analysis", type="primary"):
                with st.spinner("Analyzing transactions..."):
                    prepared = validate_and_prepare(raw, feature_order)
                    results = analyze_transactions(model, prepared, threshold)
                    st.session_state.last_results = results

                total = len(results)
                flagged = int((results["Decision"] == "REVIEW").sum())
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Analyzed", f"{total:,}")
                c2.metric("Flagged for review", f"{flagged:,}")
                c3.metric("Not flagged", f"{total - flagged:,}")
                c4.metric("Review rate", format_percent(flagged / total if total else 0))

                st.markdown("### Analysis results")
                filter_choice = st.selectbox(
                    "Filter transactions",
                    ["All transactions", "REVIEW only", "NO REVIEW only"],
                )
                shown = results
                if filter_choice == "REVIEW only":
                    shown = results[results["Decision"] == "REVIEW"]
                elif filter_choice == "NO REVIEW only":
                    shown = results[results["Decision"] == "NO REVIEW"]

                st.dataframe(
                    shown,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Fraud Probability": st.column_config.NumberColumn(
                            "Fraud Probability", format="%.6f"
                        ),
                        "Risk Score (%)": st.column_config.ProgressColumn(
                            "Risk Score (%)", min_value=0, max_value=100, format="%.2f%%"
                        ),
                    },
                )

                csv_data = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download analyzed results",
                    data=csv_data,
                    file_name="fraud_analysis_results.csv",
                    mime="text/csv",
                    type="primary",
                )

                st.markdown("### Risk distribution")
                chart_data = (
                    results["Risk Level"]
                    .value_counts()
                    .reindex(["Low", "Elevated", "High"], fill_value=0)
                    .rename_axis("Risk level")
                    .reset_index(name="Transactions")
                )
                st.bar_chart(chart_data, x="Risk level", y="Transactions")
        except Exception as exc:
            st.error(f"Unable to process this CSV: {exc}")
            st.caption(
                "Check that the file has all 30 required numeric features, "
                "with no blank or non-numeric values."
            )

# ---------------------------------------------------------------------
# About
# ---------------------------------------------------------------------
elif page == "About the Model":
    st.subheader("Model information")
    st.write(
        "This application provides a Streamlit interface for the trained credit card "
        "fraud classification model. It scores transactions using the model artifact "
        "from your project."
    )
    left, right = st.columns(2)
    with left:
        st.markdown("#### Input features")
        st.write("30 numeric features: `Time`, `V1`–`V28`, and `Amount`.")
        st.markdown("#### Target meaning")
        st.write("The original dataset uses `Class = 0` for legitimate transactions and `Class = 1` for fraud.")
    with right:
        st.markdown("#### Decision policy")
        st.metric("Configured threshold", f"{threshold:.6f}")
        st.write("Probability ≥ threshold → `REVIEW`")
        st.write("Probability < threshold → `NO REVIEW`")
    st.markdown("#### Important limitations")
    st.markdown(
        """
        - The model's output is a risk score, not a guarantee.
        - V1–V28 are anonymized transformed features; this app cannot recreate them
          from a card number or ordinary customer information.
        - Input data must use the same feature engineering and scaling approach
          used during model training.
        - A `REVIEW` decision should be investigated by an authorized analyst.
        - Protect uploaded transaction data and follow applicable privacy/security rules.
        """
    )

st.divider()
st.markdown(
    '<p class="small-note">Sentinel Card Fraud Intelligence · Model-assisted decision support · '
    "Do not use model output as the sole basis for adverse customer decisions.</p>",
    unsafe_allow_html=True,
)
