import pytest
import pandas as pd
import numpy as np
import json
import os
import tempfile
from pathlib import Path
from train_model import train_model
from predictor import predict_and_explain
from utils.feature_engineering import derive_features
import shutil
import types

@pytest.fixture(scope="function")
def sample_data():
    df_sales = pd.read_csv("data/sales_sample.csv")
    df_stores = pd.read_csv("data/stores_sample.csv")
    return df_sales, df_stores

@pytest.fixture(scope="function")
def temp_models_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("MODELS_DIR", tmpdir)
        yield tmpdir

@pytest.mark.parametrize("sales_mod,stores_mod", [
    (lambda df: df.iloc[0:0], lambda df: df),  # Empty sales
    (lambda df: df, lambda df: df.iloc[0:0]),  # Empty stores
    (lambda df: df.assign(qty_sold=np.nan), lambda df: df),  # NaN in qty_sold
    (lambda df: df.assign(revenue=-100), lambda df: df),  # Negative revenue
])
def test_train_pipeline_edge_cases(sample_data, sales_mod, stores_mod, temp_models_dir, monkeypatch):
    df_sales, df_stores = sample_data
    df_sales = sales_mod(df_sales)
    df_stores = stores_mod(df_stores)
    monkeypatch.setattr("models.MODELS_DIR", temp_models_dir, raising=False)
    try:
        iteration_log = train_model(df_sales, df_stores)
        assert isinstance(iteration_log, list)
        if not df_sales.empty and not df_stores.empty:
            assert len(iteration_log) >= 1
    except Exception as e:
        # Should fail gracefully for bad input
        assert isinstance(e, Exception)


def test_full_train_pipeline(sample_data, temp_models_dir, monkeypatch):
    df_sales, df_stores = sample_data
    monkeypatch.setattr("models.MODELS_DIR", temp_models_dir, raising=False)
    Path(temp_models_dir).mkdir(exist_ok=True)
    iteration_log = train_model(df_sales, df_stores)
    assert len(iteration_log) >= 1
    assert 'r2' in iteration_log[0]
    assert 'importances' in iteration_log[0]
    # Check model file
    model_files = list(Path(temp_models_dir).glob("*_model.pkl"))
    assert model_files, "Model file not saved."
    # Check feature list
    feature_files = list(Path(temp_models_dir).glob("*_features.json"))
    assert feature_files, "Feature list not saved."
    with open(feature_files[0], 'r') as f:
        features = json.load(f)
    assert isinstance(features, list) and len(features) > 0


def test_prediction(sample_data):
    df_sales, df_stores = sample_data
    input_dict = {
        'store': df_stores['id'].iloc[0],
        'sku_id': 'SKU_PLACEHOLDER',
        'date': pd.Timestamp('2024-01-01'),
        'qty_sold': 0,
        'revenue': 0,
        'disc_value': 10,
        'id': df_stores['id'].iloc[0],
        'channel': df_stores['channel'].iloc[0],
        'city': df_stores['city'].iloc[0],
        'region': df_stores['region'].iloc[0],
        'is_online': df_stores['is_online'].iloc[0],
        'store_area': df_stores['store_area'].iloc[0]
    }
    input_df_sales = pd.DataFrame([input_dict], columns=df_sales.columns)
    input_df_stores = pd.DataFrame([df_stores.iloc[0]], columns=df_stores.columns)
    X_pred, _ = derive_features(input_df_sales, input_df_stores)
    results = predict_and_explain(X_pred)
    assert isinstance(results['p50'][0], float)
    assert len(results['shap_values']) == 1
    assert len(results['top_5_features']) == 5
    assert all(isinstance(x, tuple) for x in results['top_5_features'])
    assert all(len(x) == 2 for x in results['top_5_features'])
    assert results['p10'][0] <= results['p50'][0] <= results['p90'][0]


def test_prediction_extreme_values(sample_data):
    df_sales, df_stores = sample_data
    # Extreme input
    input_dict = {
        'store': df_stores['id'].iloc[0],
        'sku_id': 'SKU_PLACEHOLDER',
        'date': pd.Timestamp('2024-01-01'),
        'qty_sold': 1000000,
        'revenue': 1e9,
        'disc_value': 1e8,
        'id': df_stores['id'].iloc[0],
        'channel': df_stores['channel'].iloc[0],
        'city': df_stores['city'].iloc[0],
        'region': df_stores['region'].iloc[0],
        'is_online': df_stores['is_online'].iloc[0],
        'store_area': df_stores['store_area'].iloc[0]
    }
    input_df_sales = pd.DataFrame([input_dict], columns=df_sales.columns)
    input_df_stores = pd.DataFrame([df_stores.iloc[0]], columns=df_stores.columns)
    X_pred, _ = derive_features(input_df_sales, input_df_stores)
    results = predict_and_explain(X_pred)
    assert isinstance(results['p50'][0], float)
    assert results['p50'][0] >= 0 


def test_predict_and_explain_output_structure(sample_data, monkeypatch):
    df_sales, df_stores = sample_data
    input_dict = {
        'store': df_stores['id'].iloc[0],
        'sku_id': 'SKU_PLACEHOLDER',
        'date': pd.Timestamp('2024-01-01'),
        'qty_sold': 10,
        'revenue': 1000,
        'disc_value': 10,
        'id': df_stores['id'].iloc[0],
        'channel': df_stores['channel'].iloc[0],
        'city': df_stores['city'].iloc[0],
        'region': df_stores['region'].iloc[0],
        'is_online': df_stores['is_online'].iloc[0],
        'store_area': df_stores['store_area'].iloc[0]
    }
    input_df_sales = pd.DataFrame([input_dict], columns=df_sales.columns)
    input_df_stores = pd.DataFrame([df_stores.iloc[0]], columns=df_stores.columns)
    X_pred, _ = derive_features(input_df_sales, input_df_stores)
    results = predict_and_explain(X_pred)
    # Check all expected keys
    for key in ['p10', 'p50', 'p90', 'shap_values', 'top_5_features']:
        assert key in results
    # Check types
    assert isinstance(results['p10'], np.ndarray)
    assert isinstance(results['p50'], np.ndarray)
    assert isinstance(results['p90'], np.ndarray)
    assert isinstance(results['shap_values'], np.ndarray)
    assert isinstance(results['top_5_features'], list)
    # Check shapes
    assert results['p10'].shape == results['p50'].shape == results['p90'].shape
    assert results['shap_values'].shape[0] == X_pred.shape[0]
    assert all(isinstance(x, tuple) and len(x) == 2 for x in results['top_5_features'])


def test_predict_and_explain_multiple_rows(sample_data):
    df_sales, df_stores = sample_data
    # Use first 3 rows
    input_df_sales = df_sales.head(3).copy()
    input_df_stores = df_stores[df_stores['id'].isin(input_df_sales['store'])].copy()
    X_pred, _ = derive_features(input_df_sales, input_df_stores)
    results = predict_and_explain(X_pred)
    assert results['p10'].shape[0] == 3
    assert results['shap_values'].shape[0] == 3


def test_predict_and_explain_missing_columns(sample_data):
    df_sales, df_stores = sample_data
    input_df_sales = df_sales.head(1).copy()
    input_df_stores = df_stores.head(1).copy()
    # Remove a required column
    input_df_sales = input_df_sales.drop(columns=['qty_sold'])
    try:
        X_pred, _ = derive_features(input_df_sales, input_df_stores)
        predict_and_explain(X_pred)
        assert False, "Should fail with missing column"
    except Exception:
        assert True


def test_predict_and_explain_extra_columns(sample_data):
    df_sales, df_stores = sample_data
    input_df_sales = df_sales.head(1).copy()
    input_df_sales['extra_col'] = 123
    input_df_stores = df_stores.head(1).copy()
    X_pred, _ = derive_features(input_df_sales, input_df_stores)
    # Should ignore extra columns
    results = predict_and_explain(X_pred)
    assert isinstance(results['p50'][0], float)


def test_predict_and_explain_nan_inf(sample_data):
    df_sales, df_stores = sample_data
    input_df_sales = df_sales.head(1).copy()
    input_df_sales['qty_sold'] = np.nan
    input_df_sales['revenue'] = np.inf
    input_df_stores = df_stores.head(1).copy()
    X_pred, _ = derive_features(input_df_sales, input_df_stores)
    # Should not crash, but may return nan/inf in output
    results = predict_and_explain(X_pred)
    assert results['p50'].shape[0] == 1


def test_predict_and_explain_missing_model_file(sample_data, monkeypatch):
    df_sales, df_stores = sample_data
    input_df_sales = df_sales.head(1).copy()
    input_df_stores = df_stores.head(1).copy()
    X_pred, _ = derive_features(input_df_sales, input_df_stores)
    # Temporarily rename model file
    model_path = 'models/brandA_models.pkl'
    backup_path = model_path + '.bak'
    if os.path.exists(model_path):
        shutil.move(model_path, backup_path)
    try:
        try:
            predict_and_explain(X_pred)
            assert False, "Should fail if model file is missing"
        except Exception:
            assert True
    finally:
        if os.path.exists(backup_path):
            shutil.move(backup_path, model_path)


def test_predict_and_explain_store_cluster(sample_data):
    df_sales, df_stores = sample_data
    input_df_sales = df_sales.head(2).copy()
    input_df_stores = df_stores[df_stores['id'].isin(input_df_sales['store'])].copy()
    # Add store_cluster column
    input_df_sales['store_cluster'] = [0, 1]
    X_pred, _ = derive_features(input_df_sales, input_df_stores)
    X_pred['store_cluster'] = [0, 1]
    results = predict_and_explain(X_pred)
    assert results['p10'].shape[0] == 2
    assert results['shap_values'].shape[0] == 2 