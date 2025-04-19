# /modeling/predictor.py

import logging
import pandas as pd
import numpy as np
import joblib
import os
import json
import sys
import datetime
import pytz
from typing import Optional, Tuple, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

# Add the project root directory to PYTHONPATH dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config
from modeling.trainer import MODEL_DIR, MODEL_FILENAME, MODEL_METADATA_FILENAME # Import constants
# from .. import config
# from .trainer import MODEL_DIR, MODEL_FILENAME, MODEL_METADATA_FILENAME

log = logging.getLogger(__name__)

# --- Model Loading ---
_loaded_model = None
_model_metadata = None

def load_model_and_metadata() -> Optional[Tuple[Any, Dict[str, Any]]]:
    """
    Loads the saved machine learning model and its metadata.
    Uses a singleton pattern to load only once.

    Returns:
        Optional[Tuple[Any, Dict[str, Any]]]: The loaded model object and its metadata dictionary,
                                              or None if loading fails.
    """
    global _loaded_model, _model_metadata
    if _loaded_model is None:
        model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
        metadata_path = os.path.join(MODEL_DIR, MODEL_METADATA_FILENAME)

        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            log.error(f"Model file '{model_path}' or metadata file '{metadata_path}' not found. Cannot load model.")
            return None

        try:
            log.info(f"Loading model from: {model_path}")
            _loaded_model = joblib.load(model_path)
            log.info(f"Model loaded successfully.")

            log.info(f"Loading metadata from: {metadata_path}")
            with open(metadata_path, 'r') as f:
                _model_metadata = json.load(f)
            log.info(f"Metadata loaded successfully. Model trained on {len(_model_metadata.get('feature_names', []))} features.")

        except Exception as e:
            log.error(f"Error loading model or metadata: {e}", exc_info=True)
            _loaded_model = None
            _model_metadata = None
            return None

    # Ensure we don't return a tuple with None values to match the declared return type
    if _loaded_model is None or _model_metadata is None:
        return None
        
    return (_loaded_model, _model_metadata)

# --- Feature Generation with Retry ---

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_features_for_prediction(symbol: str, end_time_utc: datetime.datetime, history_duration: pd.Timedelta) -> Optional[pd.DataFrame]:
    """
    Generates features for prediction with retry logic.

    Args:
        symbol (str): The cryptocurrency symbol (e.g., 'BTCUSDT').
        end_time_utc (datetime.datetime): The timestamp for which features are needed (timezone-aware UTC).
        history_duration (pd.Timedelta): How far back to fetch raw data for calculating features.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing features for prediction,
                                or None if feature generation fails.
    """
    from feature_engineering import feature_generator
    return feature_generator.generate_features_for_symbol(
        symbol=symbol,
        end_time_utc=end_time_utc,
        history_duration=history_duration
    )

# --- Prediction Function ---

def make_prediction(latest_features_df: pd.DataFrame) -> Optional[Tuple[int, float]]:
    """
    Makes a prediction using the loaded model on the latest feature data.

    Args:
        latest_features_df (pd.DataFrame): A DataFrame containing the most recent
                                           feature set for a single timestamp/symbol.
                                           Must contain the columns the model was trained on.

    Returns:
        Optional[Tuple[int, float]]: A tuple containing:
                                     - prediction (int): The predicted class (0 or 1).
                                     - confidence (float): The probability of the predicted class 1.
                                     Returns None if prediction fails.
    """
    model_info = load_model_and_metadata()
    if not model_info:
        log.error("Model not loaded. Cannot make prediction.")
        return None

    model, metadata = model_info
    required_features = metadata.get('feature_names')

    if not required_features:
        log.error("Model metadata does not contain feature names. Cannot ensure feature consistency.")
        return None

    if latest_features_df.empty:
        log.error("Received empty DataFrame for prediction.")
        return None

    try:
        # Select only the required features
        features_for_prediction = latest_features_df[required_features]

        # Check for NaNs in the input features for this prediction step
        if features_for_prediction.isnull().values.any():
            nan_cols = features_for_prediction.columns[features_for_prediction.isnull().any()].tolist()
            log.warning(f"NaN values detected in features for prediction: {nan_cols}. Attempting imputation with 0.")
            features_for_prediction = features_for_prediction.fillna(0)

        # Ensure only one row is passed for prediction
        if len(features_for_prediction) > 1:
            log.warning(f"Prediction input DataFrame has {len(features_for_prediction)} rows. Using the last row.")
            features_for_prediction = features_for_prediction.tail(1)

        log.debug(f"Making prediction on features: {features_for_prediction.columns.tolist()}")

        # Predict probability (returns array of shape [n_samples, n_classes])
        probabilities = model.predict_proba(features_for_prediction)

        # Safely extract the probability of class 1 (avoid tuple indexing that causes Pylance errors)
        if isinstance(probabilities, np.ndarray) and probabilities.ndim == 2:
            # For typical scikit-learn models
            prob_class_1 = float(probabilities[0, 1])  # Use comma instead of bracket indexing
        else:
            # Fallback for other model types
            prob_class_1 = float([p[1] for p in probabilities][0])

        # Get predicted class (0 or 1)
        prediction = int(model.predict(features_for_prediction)[0])

        log.info(f"Prediction successful. Class: {prediction}, Probability(Class 1): {prob_class_1:.4f}")
        return prediction, prob_class_1

    except KeyError as e:
        log.error(f"Feature mismatch during prediction. Missing feature: {e}. Required: {required_features}", exc_info=True)
        return None
    except Exception as e:
        log.error(f"Error during prediction: {e}", exc_info=True)
        return None


# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # Setup basic logging for direct script run
    logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s [%(name)s] %(message)s')

    print("--- Testing Model Predictor ---")

    # --- Prerequisites ---
    # 1. Ensure a model has been trained and saved by trainer.py
    # 2. Need feature data for prediction. We can try generating features for the latest point.

    # Attempt to load model and metadata
    model_info = load_model_and_metadata()
    if not model_info:
        print("Model or metadata not found or failed to load. Run trainer.py first.")
    else:
        _, metadata = model_info
        print("Model and metadata loaded.")

        # --- Generate Sample Features for Prediction ---
        # In a real run, this would come from feature_generator using the latest available data
        print("\nGenerating sample features for prediction (using latest data)...")
        test_symbol = 'DOGEUSDT' # Use a symbol likely trained on
        # Use current time to get the very latest features possible
        pred_time = datetime.datetime.now(pytz.utc)
        # Need enough history to calculate lags/rolling features for the *single* prediction point
        hist_for_pred = pd.Timedelta(days=3) # Adjust as needed based on longest lookback

        try:
            latest_features = generate_features_for_prediction(
                symbol=test_symbol,
                end_time_utc=pred_time,
                history_duration=hist_for_pred
            )

            if latest_features is not None and not latest_features.empty:
                # Get the single most recent row of features
                prediction_input_df = latest_features.tail(1)
                print(f"Generated {len(prediction_input_df)} row(s) of features for prediction.")
                # print(prediction_input_df) # Optionally print the feature row

                # --- Make Prediction ---
                prediction_result = make_prediction(prediction_input_df)

                if prediction_result:
                    pred_class, pred_prob = prediction_result
                    print(f"\nPrediction for {test_symbol} at ~{pred_time}:")
                    print(f"  Predicted Class: {pred_class} (0=DOWN/SIDEWAYS, 1=UP)")
                    print(f"  Probability (Class 1): {pred_prob:.4f}")
                else:
                    print("\nFailed to make prediction. Check logs.")
            else:
                print("\nFailed to generate features for prediction input. Cannot test predictor.")
        except Exception as e:
            print(f"\nError during feature generation or prediction: {e}")

    print("\n--- Test Complete ---")

