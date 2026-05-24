# CRIS Dashboard

CRIS is an upload-first Streamlit dashboard for the ChurnZero banking churn problem.

The app does not load any private CSV by default. Users must upload a dataset at runtime.

## Upload Logic

- If the uploaded CSV already has `churn_prediction` and `churn_probability`, the app directly shows analytics.
- If the uploaded CSV does not have predictions, the app uses `preprocessor.pkl` and `xgboost_churn_model.json` to predict first, then shows analytics.

## Run Locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Important



Keep these model artifacts in the repo:

- `preprocessor.pkl`
- `xgboost_churn_model.json`
- `app_schema.json`

