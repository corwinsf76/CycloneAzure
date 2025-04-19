# /modeling/trainer.py

import sys
import os
# Get the directory containing this script (modeling/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (cyclonev2/)
project_root = os.path.dirname(current_dir)
# Add the project root to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"DEBUG: Added project root to sys.path: {project_root}") # Optional debug print

import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split # Use time-series split later
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
import joblib # For saving/loading model
import datetime
import pytz
from typing import List, Optional, Tuple, Dict, Any
import time
# os was already imported above
import json # Added json import for metadata
from tenacity import retry, stop_after_attempt, wait_exponential

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config
from database import db_utils # To potentially log results or fetch symbols
from feature_engineering import feature_generator # To generate features
# Need binance_client if fetching symbols dynamically in __main__ block
from data_collection import binance_client
# from .. import config
# from ..database import db_utils
# from ..feature_engineering import feature_generator
# from ..data_collection import binance_client


log = logging.getLogger(__name__)

# Define paths for saving model artifacts
MODEL_DIR = "trained_models" # Directory to save models (create if not exists)
MODEL_FILENAME = "lgbm_classifier_v1.joblib"
MODEL_METADATA_FILENAME = "lgbm_classifier_v1_metadata.json" # Store feature list, metrics etc.

# --- Data Loading and Preprocessing ---

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def load_training_data(symbols: List[str], end_time_utc: datetime.datetime, history_duration: pd.Timedelta) -> Optional[pd.DataFrame]:
    """
    Loads features for multiple symbols over a historical period for training with retry logic.

    Args:
        symbols (List[str]): List of symbols to generate features for.
        end_time_utc (datetime.datetime): The end timestamp for the training data period (timezone-aware UTC).
        history_duration (pd.Timedelta): How far back the training data should span.

    Returns:
        Optional[pd.DataFrame]: A concatenated DataFrame of features for all symbols,
                                or None if data generation fails for all symbols.
    """
    all_features_df = []
    log.info(f"Loading training data for {len(symbols)} symbols up to {end_time_utc}...")

    for symbol in symbols:
        log.debug(f"Generating features for training: {symbol}")
        features_df = feature_generator.generate_features_for_symbol(
            symbol=symbol,
            end_time_utc=end_time_utc,
            history_duration=history_duration
        )
        if features_df is not None and not features_df.empty:
            all_features_df.append(features_df)
        else:
            log.warning(f"Could not generate features for symbol {symbol}. Skipping.")

    if not all_features_df:
        log.error("Failed to generate features for any symbols. Cannot proceed with training.")
        return None

    combined_df = pd.concat(all_features_df, ignore_index=True)
    log.info(f"Combined training data shape: {combined_df.shape}")

    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return combined_df


def preprocess_data(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """
    Preprocesses the feature DataFrame for training.
    Handles NaNs, separates features and target, performs train/test split.

    Args:
        df (pd.DataFrame): The combined feature DataFrame.
        target_col (str): The name of the target variable column.

    Returns:
        Tuple containing:
        - X_train (pd.DataFrame): Training features.
        - y_train (pd.Series): Training target.
        - X_test (pd.DataFrame): Testing features.
        - y_test (pd.Series): Testing target.
        - feature_names (List[str]): List of feature column names used.
    """
    log.info("Preprocessing data...")

    # Drop rows where target is NaN (essential)
    df.dropna(subset=[target_col], inplace=True)
    log.info(f"Shape after dropping NaN targets: {df.shape}")

    if df.empty:
        raise ValueError("No valid data remaining after dropping NaN targets.")

    # Prepare features
    feature_names, X = _prepare_features(df)
    y = df[target_col]

    log.info(f"Number of features: {len(feature_names)}")
    log.debug(f"Feature names: {feature_names[:10]}...") # Log first few

    # --- Handle NaNs in Features ---
    # Option 1: Drop rows with any NaNs in features
    # initial_len = len(X)
    # X = X.dropna()
    # y = y.loc[X.index]
    # log.info(f"Dropped {initial_len - len(X)} rows due to NaNs in features.")

    # Option 2: Imputation (e.g., fill with mean, median, or 0) - Often preferred for tree models
    # Check percentage of NaNs per column
    nan_perc = X.isnull().sum() / len(X) * 100
    cols_with_nan = nan_perc[nan_perc > 0]
    if not cols_with_nan.empty:
        log.warning(f"Features contain NaNs. Imputing with 0. NaN percentages:\n{cols_with_nan}")
        # Simple imputation with 0 (LightGBM can handle NaNs internally too, but explicit is safer)
        X = X.fillna(0)
    else:
        log.info("No NaNs found in feature columns.")

    if X.empty or y.empty:
        raise ValueError("No data remaining after NaN handling.")

    # --- Train/Test Split ---
    # IMPORTANT: For time series, MUST split chronologically, not randomly!
    # Train on older data, test on newer data to simulate real-world prediction.
    test_size = 0.2 # e.g., use latest 20% of data for testing
    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    log.info(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    log.info(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
    log.info(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")

    return X_train, y_train, X_test, y_test, feature_names


def _prepare_features(df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    """Prepares features for model training, including new API data features."""
    
    # Base features (existing technical indicators and price data)
    base_features = [col for col in df.columns if col.startswith(('lag_', 'roll_', 'sma_', 'ema_', 'rsi_', 'macd_'))]
    
    # Sentiment features
    sentiment_features = [col for col in df.columns if col.startswith(('sent_', 'cp_'))]
    
    # New API features
    alphavantage_features = [col for col in df.columns if col.startswith('av_')]
    coingecko_features = [col for col in df.columns if col.startswith('cg_')]
    
    # Time features
    time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    
    # Combine all features
    feature_columns = (base_features + sentiment_features + alphavantage_features + 
                      coingecko_features + time_features)
    
    # Create feature importance groups for later analysis
    feature_groups = {
        'price_technical': base_features,
        'sentiment': sentiment_features,
        'health_metrics': alphavantage_features,
        'market_metrics': coingecko_features,
        'time': time_features
    }
    
    # Store feature groups in metadata
    metadata = {
        'feature_groups': feature_groups,
        'total_features': len(feature_columns)
    }
    
    # Save metadata
    with open('trained_models/feature_metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    return feature_columns, df[feature_columns]


# --- Model Training and Evaluation ---

def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
                X_test: pd.DataFrame, y_test: pd.Series,
                model_params: Optional[Dict[str, Any]] = None) -> Tuple[LGBMClassifier, Dict[str, Any]]:
    """
    Trains the model with enhanced features and returns performance metrics.
    
    Args:
        X_train: Training feature DataFrame
        y_train: Training target Series
        X_test: Testing feature DataFrame
        y_test: Testing target Series
        model_params: Optional dictionary of model parameters to override defaults
        
    Returns:
        Tuple containing the trained model and a dictionary of performance metrics
    """
    if model_params is None:
        model_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 32,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
    
    # Initialize and train model
    model = LGBMClassifier(**model_params)
    
    # Store LightGBM-specific parameters separately to avoid type errors
    fit_params = {
        'eval_set': [(X_test, y_test)],
        'verbose': False
    }
    
    # early_stopping_rounds is a valid LightGBM parameter
    # but we'll set it separately to avoid Pylance errors
    fit_params['early_stopping_rounds'] = 50
    
    model.fit(
        X_train, y_train,
        **fit_params
    )
    
    # Calculate predictions and metrics
    y_pred = model.predict(X_test)
    
    # Get probabilities safely with robust type checking
    probabilities = model.predict_proba(X_test)
    
    # Handle predicted probabilities with proper type checking and shape validation
    prob_class_1 = None
    if isinstance(probabilities, np.ndarray) and probabilities.ndim == 2 and probabilities.shape[1] > 1:
        # Direct array access for proper numpy array
        prob_class_1 = probabilities[:, 1]
    else:
        # Convert list-like structure to numpy array with proper type safety
        try:
            # Convert to list first to handle various array-like structures
            probs_list = list(probabilities)
            # Create a properly shaped numpy array
            prob_class_1 = np.array([
                p[1] if isinstance(p, (list, np.ndarray)) and len(p) > 1 else 0.0 
                for p in probs_list
            ])
        except (TypeError, IndexError) as e:
            log.warning(f"Error processing probabilities, using zeros: {e}")
            prob_class_1 = np.zeros(len(y_test))
    
    # Convert predicted values to ensure correct types
    y_pred_array = np.array(y_pred)
    y_test_array = np.array(y_test)
    
    # Ensure binary target values for classification metrics
    # Use astype to explicitly convert to the expected types for the metrics functions
    y_pred_array = y_pred_array.astype(int)
    y_test_array = y_test_array.astype(int)
    
    metrics = {
        'accuracy': float(accuracy_score(y_test_array, y_pred_array)),
        'precision': float(precision_score(y_test_array, y_pred_array, zero_division=0)),
        'recall': float(recall_score(y_test_array, y_pred_array, zero_division=0)), 
        'f1': float(f1_score(y_test_array, y_pred_array, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_test_array, prob_class_1))
    }
    
    # Calculate feature importance by groups
    # Explicitly convert feature_importances_ to a list to avoid sparse matrix issues
    importances = model.feature_importances_
    if hasattr(importances, "toarray"):
        # Handle sparse matrices by converting to dense array first
        importances = importances.toarray().flatten()
    elif not isinstance(importances, (list, np.ndarray)):
        # Convert any other type to list
        importances = list(importances)
        
    feature_importance = pd.DataFrame({
        'feature': X_train.columns.tolist(),
        'importance': importances
    })
    
    # Create directory if it doesn't exist for feature metadata
    os.makedirs("trained_models", exist_ok=True)
    
    # Load feature groups from metadata if file exists
    try:
        # Check if metadata file exists before attempting to open it
        if os.path.exists('trained_models/feature_metadata.json'):
            with open('trained_models/feature_metadata.json', 'r') as f:
                feature_metadata = json.load(f)
                
            # Calculate importance by group
            group_importance = {}
            # Access feature_groups with a safe get() method with default value
            feature_groups = feature_metadata.get('feature_groups', {})
            for group, features in feature_groups.items():
                # Check if features is a valid list before using isin() method
                if isinstance(features, list) and features:
                    group_mask = feature_importance['feature'].isin(features)
                    if not group_mask.empty:
                        group_importance[group] = float(feature_importance.loc[group_mask, 'importance'].sum())
                    else:
                        group_importance[group] = 0.0
                else:
                    group_importance[group] = 0.0
            
            # Add group importance to metrics
            metrics['group_importance'] = group_importance
        else:
            log.warning("Feature metadata file not found, skipping group importance calculation")
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        log.warning(f"Could not load feature metadata for group importance: {e}")
    
    return model, metrics


def save_model(model: LGBMClassifier, features: List[str], metrics: Dict[str, Any], version: str = "v1"):
    """
    Saves the trained model and its metadata.
    """
    # Create directory if it doesn't exist
    os.makedirs("trained_models", exist_ok=True)
    
    # Save the model
    model_path = f"trained_models/lgbm_classifier_{version}.joblib"
    joblib.dump(model, model_path)
    
    # Prepare metadata
    metadata = {
        'version': version,
        'features': features,
        'metrics': metrics,
        'training_date': datetime.datetime.now(pytz.utc).isoformat(),
        'model_params': model.get_params(),
        'feature_importances': dict(zip(features, model.feature_importances_.tolist()))
    }
    
    # Save metadata
    metadata_path = f"trained_models/lgbm_classifier_{version}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


# --- Main Training Orchestration ---
if __name__ == '__main__':
    # Setup basic logging for direct script run (if not already configured by main.py)
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s [%(name)s] %(message)s')

    print("--- Running Model Trainer ---")

    # --- Parameters ---
    # Define symbols to train on
    symbols_to_train = [] # Start with empty list

    if not symbols_to_train:
        # Fetch dynamically using binance_client
        log.info("No symbols specified, attempting to fetch target symbols...")
        try:
            # Ensure binance_client is imported
            symbols_to_train = binance_client.get_target_symbols()
            log.info(f"Fetched {len(symbols_to_train)} target symbols for training.")
        except Exception as e:
            log.error(f"Failed to fetch target symbols: {e}", exc_info=True)
            symbols_to_train = [] # Ensure it's an empty list if fetch fails

    if not symbols_to_train:
        log.critical("No symbols available to train on (fetch failed or returned empty). Exiting.")
        sys.exit(1) # Exit if no symbols could be determined

    # Ensure symbols_to_train is a list of strings (get_target_symbols should return this)
    if not isinstance(symbols_to_train, list) or not all(isinstance(s, str) for s in symbols_to_train):
        log.critical(f"Symbols to train is not a list of strings: {symbols_to_train}. Exiting.")
        sys.exit(1)

    log.info(f"Proceeding to train with symbols: {symbols_to_train[:20]}...") # Log first few symbols

    training_end_time = datetime.datetime.now(pytz.utc) - pd.Timedelta(hours=1) # Train up to 1 hour ago
    training_history = pd.Timedelta(days=90) # Use last 90 days for training data
    target_variable_name = f'target_up_{config.PREDICTION_HORIZON_PERIODS}p'

    # --- Pipeline ---
    try:
        # 1. Load Data
        training_df = load_training_data(symbols_to_train, training_end_time, training_history)

        if training_df is not None and not training_df.empty:
            # 2. Preprocess Data
            X_train, y_train, X_test, y_test, features = preprocess_data(training_df, target_variable_name)

            # 3. Train Model
            trained_model, test_metrics = train_model(X_train, y_train, X_test, y_test)

            # 4. Save Model
            if trained_model and test_metrics:
                save_model(trained_model, features, test_metrics)
                log.info("Model training and saving process completed successfully.")
            else:
                log.error("Model training failed during train/evaluate step.")
        else:
            log.error("Failed to load training data (returned None or empty DataFrame).")

    except ValueError as ve:
        log.critical(f"ValueError during training pipeline: {ve}", exc_info=True)
    except Exception as e:
        log.critical(f"An unexpected error occurred during the training pipeline: {e}", exc_info=True)

    print("\n--- Training Script Finished ---")