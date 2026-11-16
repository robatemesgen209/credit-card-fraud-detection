# Suggested demonstration video (under 10 minutes)

1. Introduce the finance use case: prioritize suspicious card transactions for human review.
2. Show the project structure and explain the Module 5 hand-off: cost-sensitive XGBoost plus preprocessing and threshold.
3. Run `python -m src.train` and show the saved artifact and `reports/evaluation.json`.
4. Run tests with `pytest -q`.
5. Start the API with `uvicorn src.api:app --reload`; open `/docs`, call `/health`, then score a transaction using all 30 features.
6. Start the UI with `streamlit run app.py`; demonstrate single-row and CSV scoring.
7. Explain precision/recall trade-off, historical dataset limitations, and human-review safeguards.
