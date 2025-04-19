# /modeling/trainer.py

import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split # Use time-series split later
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib # For saving/loading model
import datetime
import pytz
from typing import List, Optional, Tuple, Dict, Any # <--- Added Dict and Any here
import time
import os # Added os import for makedirs
import json # Added json import for metadata

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

def load_training_data(symbols: List[str], end_time_utc: datetime.datetime, history_duration: pd.Timedelta) -> Optional[pd.DataFrame]:
    """
    Loads features for multiple symbols over a historical period for training.

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
        # Ensure end_time aligns with candle close if needed by feature generator logic
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

    # Concatenate features from all symbols
    combined_df = pd.concat(all_features_df, ignore_index=True)
    log.info(f"Combined training data shape: {combined_df.shape}")

    # Optional: Handle potential infinite values if not handled earlier
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

    # Define features (X) and target (y)
    # Exclude non-feature columns like 'symbol', 'open_time' (index), and the target itself
    non_feature_cols = [target_col, 'symbol'] # Add others if needed
    # Ensure index is not accidentally included as a feature if it was reset
    feature_names = [col for col in df.columns if col not in non_feature_cols and col != df.index.name and df.dtypes[col] in [np.int64, np.float64, np.int32, np.float32]]
    X = df[feature_names]
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


# --- Model Training and Evaluation ---

def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Optional[Tuple[lgb.LGBMClassifier, Dict]]:
    """
    Trains a LightGBM classification model.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Testing data for evaluation.

    Returns:
        Optional[Tuple[lgb.LGBMClassifier, Dict]]: The trained LightGBM model and test metrics,
                                                   or None if training fails.
    """
    log.info("Training LightGBM model...")

    # --- Model Parameters ---
    # TODO: Tune these parameters using techniques like GridSearchCV or Optuna
    params = {
        'objective': 'binary', # Binary classification (UP/DOWN)
        'metric': 'auc', # Area Under Curve - good for imbalanced classes
        'boosting_type': 'gbdt',
        'n_estimators': 1000, # Number of boosting rounds
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1, # No limit
        'seed': 42,
        'n_jobs': -1, # Use all available cores
        'colsample_bytree': 0.8, # Feature subsampling
        'subsample': 0.8, # Data subsampling
        'reg_alpha': 0.1, # L1 regularization
        'reg_lambda': 0.1, # L2 regularization
        # Add is_unbalance=True or scale_pos_weight if classes are very imbalanced
    }

    model = lgb.LGBMClassifier(**params)

    try:
        # Train the model with early stopping
        eval_set = [(X_test, y_test)]
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True)]

        start_time = time.time()
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric='auc', # Evaluate on AUC during training
            callbacks=callbacks
        )
        end_time = time.time()
        log.info(f"Model training completed in {end_time - start_time:.2f} seconds.")

        # --- Evaluation ---
        log.info("Evaluating model performance...")
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_prob_test = model.predict_proba(X_test)[:, 1] # Probability of class 1 (UP)

        log.info("--- Training Set Performance ---")
        log.info(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
        log.info(f"AUC: {roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]):.4f}")
        log.info(f"\n{classification_report(y_train, y_pred_train)}")

        log.info("--- Test Set Performance ---")
        log.info(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
        log.info(f"AUC: {roc_auc_score(y_test, y_prob_test):.4f}")
        log.info(f"\n{classification_report(y_test, y_pred_test)}")
        log.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_test)}")

        # Store performance metrics
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'auc': roc_auc_score(y_test, y_prob_test),
            'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
        }

        return model, test_metrics

    except Exception as e:
        log.error(f"Error during model training or evaluation: {e}", exc_info=True)
        return None, None


def save_model(model: lgb.LGBMClassifier, feature_names: List[str], metrics: Dict):
    """Saves the trained model and metadata."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)

        model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
        metadata_path = os.path.join(MODEL_DIR, MODEL_METADATA_FILENAME)

        log.info(f"Saving trained model to: {model_path}")
        joblib.dump(model, model_path)

        # Save metadata (features used, performance metrics)
        metadata = {
            'model_name': MODEL_FILENAME,
            'training_timestamp_utc': datetime.datetime.now(pytz.utc).isoformat(),
            'feature_names': feature_names,
            'test_set_metrics': metrics,
            'training_params': model.get_params() # Store model parameters used
        }

        log.info(f"Saving model metadata to: {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    except Exception as e:
        log.error(f"Error saving model or metadata: {e}", exc_info=True)


# --- Main Training Orchestration ---
if __name__ == '__main__':
    # Setup basic logging for direct script run
    logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s [%(name)s] %(message)s')

    print("--- Running Model Trainer ---")

    # --- Parameters ---
    # Define symbols to train on (fetch dynamically or use a fixed list)
    # Example: Get symbols from config or dynamically fetch target symbols
    # symbols_to_train = config.SYMBOLS_TO_MONITOR # Assuming this exists in config
    symbols_to_train = ['BTCUSDT', 'ETHUSDT'] # Example fixed list for testing
    if not symbols_to_train:
         # Fallback or fetch dynamically
         log.info("No symbols specified, attempting to fetch target symbols...")
         # Ensure binance_client is imported if using this
         symbols_to_train = binance_client.get_target_symbols()

    if not symbols_to_train:
        log.critical("No symbols available to train on. Exiting.")
        exit()

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
                log.error("Model training failed.")
        else:
            log.error("Failed to load training data.")

    except Exception as e:
        log.critical(f"An error occurred during the training pipeline: {e}", exc_info=True)

    print("\n--- Training Script Finished ---")
