# Credit Card Fraud Review — Module 6

**Domain:** Finance  
**Project:** Detecting Fraudulent Credit Card Transactions  
**Alignment:** Module 5 Group Project Implementation Report, Group 22

This is a runnable educational prototype that scores credit-card transactions and returns a fraud-risk probability plus a human-review decision. It uses the Module 5 report's **cost-sensitive XGBoost** approach, not the earlier logistic-only draft.

## Module 5 workflow carried forward

- Remove exact duplicates before splitting.
- Use the 30 numeric inputs (`Time`, `Amount`, `V1`–`V28`); `Class` is the target.
- Stratified 70% train / 15% validation / 15% test split, random state 42.
- Median imputation and `RobustScaler` inside the pipeline.
- XGBoost: 250 trees, depth 5, learning rate 0.05, subsampling and column subsampling, training-set `scale_pos_weight`.
- Select the highest-precision threshold on validation data among thresholds with recall at least 0.80.
- Evaluate once on the held-out test partition.
- Serve the saved preprocessing/model artifact through FastAPI.

## Dataset and this rebuild's results

The uploaded CSV contains 284,807 raw rows, 31 columns, 1,081 exact duplicate rows, and 0 missing values. After duplicate removal it contains 283,726 rows and 473 fraud cases.

The current package's held-out test results:
- Average precision: **0.7942**
- ROC-AUC: **0.9686**
- Precision: **0.8028**
- Recall: **0.8028**
- F1: **0.8028**
- Threshold: **0.535887**
- Confusion matrix (actual rows [legitimate, fraud], predicted columns [legitimate, fraud]): `[[42474, 14], [14, 57]]`

These are the fresh Module 6 rebuild metrics, not copied from the Module 5 report. The report's original test metrics were AP 0.8021, precision 0.9804, recall 0.7042, and F1 0.8197. Differences can occur due to retraining/library versions and exact implementation details. See `reports/evaluation.json`.

## Run on Windows

Install Python 3.11 or newer, then from this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

To retrain from the included CSV:

```powershell
python -m src.train
```

Start the API:

```powershell
uvicorn src.api:app --reload
```
https://robatemesgen209-credit-card-fraud-detection-app-2noat7.streamlit.app/
API docs: http://127.0.0.1:8000/docs  
Health: http://127.0.0.1:8000/health  
Scoring: `POST http://127.0.0.1:8000/v1/fraud/score`

Request body shape:

```json
{
  "transaction": {
    "Time": 0,
    "V1": 0,
    "V2": 0,
    "V3": 0,
    "V4": 0,
    "V5": 0,
    "V6": 0,
    "V7": 0,
    "V8": 0,
    "V9": 0,
    "V10": 0,
    "V11": 0,
    "V12": 0,
    "V13": 0,
    "V14": 0,
    "V15": 0,
    "V16": 0,
    "V17": 0,
    "V18": 0,
    "V19": 0,
    "V20": 0,
    "V21": 0,
    "V22": 0,
    "V23": 0,
    "V24": 0,
    "V25": 0,
    "V26": 0,
    "V27": 0,
    "V28": 0,
    "Amount": 0
  }
}
```

Start the Streamlit interface in a second terminal:

```powershell
streamlit run app.py
```

## Docker

```powershell
docker compose up --build
```

API will be available at http://localhost:8000.

## Project files

- `data/creditcard.csv` — exact uploaded benchmark CSV
- `src/train.py` — duplicate handling, split, training, validation threshold selection, test evaluation
- `src/predict.py` — strict validation and shared inference
- `src/api.py` — FastAPI health and scoring endpoints
- `app.py` — Streamlit single and batch scoring UI
- `artifacts/fraud_model.joblib` — trained pipeline, threshold, feature order and metadata
- `reports/` — metrics and schema
- `tests/` — inference validation tests
- `RETROSPECTIVE.md`, `DATA_STATEMENT.md`, `VIDEO_SCRIPT.md`

## Responsible use

The API returns `review` or `no_review`; it does not block a card or establish that fraud occurred. The Module 5 report notes that its selected threshold missed 21 of 71 test fraud cases. The benchmark is historical, the features are anonymized, and the prototype lacks production security and monitoring. Do not connect it to live payments or real customer decisioning without institution-specific governance, security, privacy review, and prospective validation.
# credit-card-fraud-detection
