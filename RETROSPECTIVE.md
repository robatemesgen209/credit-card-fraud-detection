# Module 6 retrospective

## What was implemented
The Module 6 package carries the Module 5 cost-sensitive XGBoost model into a runnable application. It includes the preprocessing/model pipeline, a saved model artifact, strict single-transaction inference, a FastAPI endpoint, a Streamlit interface, Docker configuration, and tests.

## What was learned
The Module 5 report shows why accuracy is misleading for this dataset: a majority-class model can exceed 99.8% accuracy while detecting no fraud. Average precision, precision, recall, F1, alert rate, and confusion-matrix counts are more useful. The report's final XGBoost operating point achieved 0.8021 average precision and 0.9804 precision, but only 0.7042 recall, missing 21 of 71 test fraud cases. That threshold trade-off must remain visible in the deployment interface and documentation.

## Module 6 rebuild results
This package retrains the model from the uploaded CSV using the documented duplicate-removal and stratified 70/15/15 workflow. Its exact metrics are in `reports/evaluation.json`. Results may differ slightly from the Module 5 report because this is a fresh run and library versions/implementation details can affect fitted trees and thresholds.

## Limitations and next steps
The dataset covers only two days in 2013 and uses anonymized PCA features. The prototype is not connected to a payment network and does not implement authentication, authorization, production audit integration, rate limiting, drift monitoring, or institution-specific compliance. A production path requires chronological/prospective validation, a cost-based threshold agreed with stakeholders, probability calibration, security hardening, and current institution-specific data. Human review remains mandatory; this model must not automatically block cards.
