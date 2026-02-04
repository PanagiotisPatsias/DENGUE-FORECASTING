import pandas as pd
from pathlib import Path

from src.core.data_loader import DataLoader
from src.core.feature_engineer import FeatureEngineer
from src.core.forecaster import Forecaster

def test_data_quality():
    dengue_path = Path("data/infodengue_capitals_subsetBR.csv")
    sst_path = Path("data/sst_indices.csv")

    assert dengue_path.exists(), f"Missing dengue dataset: {dengue_path}"
    assert sst_path.exists(), f"Missing SST dataset: {sst_path}"

    dengue_df = pd.read_csv(dengue_path)
    sst_df = pd.read_csv(sst_path)

    dengue_required = {"data_iniSE", "casos_est"}
    sst_required = {"YR", "MON", "NINO1+2", "NINO3", "NINO3.4", "ANOM.3"}

    missing_dengue = dengue_required - set(dengue_df.columns)
    missing_sst = sst_required - set(sst_df.columns)

    assert not missing_dengue, f"[dengue] Missing columns: {sorted(missing_dengue)}"
    assert not missing_sst, f"[sst] Missing columns: {sorted(missing_sst)}"


def test_features_exist_and_not_all_nan():
    loader = DataLoader()
    df = loader.load_and_prepare_data()

    engineer = FeatureEngineer()
    df_feat, feature_cols = engineer.create_features(df)

    assert feature_cols, "No feature columns returned by FeatureEngineer"

    missing = [c for c in feature_cols if c not in df_feat.columns]
    assert not missing, f"Missing engineered features: {missing}"

    all_nan = [c for c in feature_cols if df_feat[c].isna().all()]
    assert not all_nan, f"All-NaN feature columns: {all_nan}"


def test_forecast_shape():
    loader = DataLoader()
    df = loader.load_and_prepare_data()

    engineer = FeatureEngineer()
    df_feat, feature_cols = engineer.create_features(df)

    forecaster = Forecaster(engineer)

    from src.core.model_trainer import ModelTrainer
    trainer = ModelTrainer()
    base_model = list(trainer.models.values())[0]

    forecast_df, _, _ = forecaster.refit_and_forecast(
        df_feat,
        feature_cols,
        base_model,
        forecast_year=2026,
        train_max_year=2025,
        exclude_years=[2024],
    )

    assert len(forecast_df) == 4, "Forecast should have 4 quarters for 2026"
    assert "year_quarter" in forecast_df.columns
    assert "predicted_casos_est" in forecast_df.columns
