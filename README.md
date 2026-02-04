# Dengue Forecasting

End-to-end dengue forecasting pipeline with model training, monitoring, and a Streamlit UI.

## What This Repo Contains
- **Data pipeline**: Loads dengue and SST data, engineers features, trains models, evaluates 2023 and 2025, and forecasts 2026.
- **Monitoring**: Logs MLflow runs, drift checks, and summary artifacts.
- **UI**: Streamlit app for training and forecasting.
- **Deploy**: GitHub Actions workflow for Cloud Run + Cloud Scheduler.
- **Quality checks**: Basic data-quality tests.

## Project Structure
- `src/` Core pipeline code (data loading, feature engineering, training, forecasting, monitoring)
- `app.py` Streamlit app local
- `data/` Input datasets (dengue + SST)
- `scripts/` Utilities (training, drift scheduler, MLflow checks, deployment helper)
- `.github/workflows/deploy.yml` CI/CD to Cloud Run + Scheduler
- `data_quality/` Data quality tests

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run The Pipeline (Local)
```bash
python -m src.main
```

## Run The Streamlit App (Local)
```bash
streamlit run app.py
```

## Drift Monitoring (Local)
Run once:
```bash
python scripts/drift_scheduler.py --run-once
```
Or run continuously:
```bash
python scripts/drift_scheduler.py --interval daily
```

## Data Quality Tests
```bash
python -m pytest -q data_quality/test_data_quality.py
```

## MLflow
By default the code uses `MLFLOW_TRACKING_URI` if set, otherwise logs locally to `./mlruns`.

## Cloud Deploy (GitHub Actions)
Workflow: `.github/workflows/deploy.yml`

On push to `main`, it:
1. Builds and pushes the Docker image to Artifact Registry
2. Creates/updates a Cloud Run Job
3. Creates/updates a Cloud Scheduler job (daily 08:00 Europe/Zurich)
4. Executes the job once after deploy


## Notes
- The Streamlit app expects dengue and SST CSVs in `data/`.
