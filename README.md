# Dengue Forecasting

End-to-end dengue forecasting pipeline with model training, monitoring and a Streamlit UI.

## What This Repo Contains
- **Data pipeline**: Loads dengue and SST data, engineers features, trains models, evaluates 2023 and 2025 and forecasts 2026.
- **Monitoring**: Logs MLflow runs, drift checks and summary artifacts.
- **UI**: Streamlit app for training and forecasting.
- **Deploy**: GitHub Actions workflow for Cloud Run + Cloud Scheduler.
- **Quality checks**: Basic data-quality tests.

## Project Structure
- `src/` Core pipeline code (data loading, feature engineering, training, forecasting, monitoring)
- `app.py` Streamlit app local
- `data/` Input datasets (dengue + SST)
- `scripts/` Utilities ( drift scheduler, MLflow checks)
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
Notes:
- Runs backtesting for 2023 and 2025, logs metrics to MLflow and performs drift check + drift alert.
- Forecasting output is printed to the terminal (forecast is not logged to MLflow in `src.main`).

Optional arguments:
```bash
python -m src.main --start-scheduler --scheduler-interval 6h
```
Or use the helper script:
```bash
./scripts/run_pipeline.sh
```
If you get "Permission denied", make the script executable:
```bash
chmod +x scripts/run_pipeline.sh
```

## Run The Streamlit App (Local)
```bash
streamlit run app.py
```
Or use the helper script to start both MLflow UI and Streamlit:
```bash
./scripts/run_app.sh
```
If you get "Permission denied", make the script executable:
```bash
chmod +x scripts/run_app.sh
```
Notes:
- From the UI, you can send backtesting (2023/2025) and the 2026 forecast to MLflow (when "Log results to MLflow" is enabled).

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

## Run MLflow (Local)
From the project root:
```bash
mlflow ui --backend-store-uri "file://$(pwd)/mlruns" --host 0.0.0.0 --port 5000
```
Then open `http://localhost:5000` in your browser.

## Run With Docker Compose
Use Docker Compose to run both MLflow and Streamlit:
```bash
docker compose up --build
```
Services:
- MLflow UI on port `5000`
- Streamlit app on port `8501`

## Windows Notes
On Windows (PowerShell/CMD), run the Python commands directly:
```bash
python -m src.main
streamlit run app.py
```
Shell scripts (`.sh`) and `chmod +x` are Linux/macOS only.

## Cloud Deploy (GitHub Actions)
Workflow: `.github/workflows/deploy.yml`

On push to `main`, it:
1. Data quality tests
2. Builds and pushes the Docker image to Artifact Registry
3. Creates/updates a Cloud Run Job
4. Creates/updates a Cloud Scheduler job (daily 08:00 Europe/Zurich)
5. Executes the job once after deploy

## Cloud Run Job (Pipeline)
The Cloud Run Job executes the training pipeline as a batch task by running:
```bash
python -m src.main
```
It is triggered manually or by Cloud Scheduler.


## Notes
- The Streamlit app expects dengue and SST CSVs in `data/`.
