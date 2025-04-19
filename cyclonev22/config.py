# /config.py
import os
import logging
from dotenv import load_dotenv
import sys # Import sys for checking handlers
import ast # Import ast for literal_eval

# --- Load .env File ---
# Load environment variables from .env file if it exists
# Searches current directory and parents.
env_path = load_dotenv()
# --- ADD DEBUG LINE HERE ---
print(f"DEBUG: dotenv loaded path: {env_path}")
# ---------------------------
if env_path:
    # Use print before logging is configured
    print(f"Loaded environment variables from: {env_path}")
else:
    # Use print before logging is configured
    print("INFO: .env file not found or empty. Relying on system environment variables.")

# Configure logging temporarily here if needed for early config errors
# logging.basicConfig(level=logging.DEBUG) # Example temporary config
log = logging.getLogger(__name__)

# --- Environment Variable Loading Function ---
def get_env_variable(var_name, default=None, required=False, var_type=str):
    """
    Retrieves an environment variable, logs warnings/errors, and performs type casting.
    Includes attempt to strip inline comments and uses literal_eval for lists.
    Refined error handling and debugging.
    """
    raw_value = os.getenv(var_name)
    print(f"DEBUG: Raw value for '{var_name}' from os.getenv: '{raw_value}' (Type: {type(raw_value)})")

    if raw_value is None:
        if required:
            print(f"CRITICAL: Required environment variable '{var_name}' is not set.")
            raise ValueError(f"Required environment variable '{var_name}' is not set.")
        else:
            print(f"Environment variable '{var_name}' not set, using default: {default}")
            return default

    value_cleaned = raw_value
    if isinstance(raw_value, str):
         value_cleaned = raw_value.split('#')[0].strip()
         if value_cleaned != raw_value:
              print(f"DEBUG: Cleaned value for '{var_name}': '{value_cleaned}'")

    if not value_cleaned: # Handle empty string after stripping
         if default is not None:
              print(f"Warning: Value for '{var_name}' became empty after stripping comment, using default: {default}")
              return default
         elif required:
              raise ValueError(f"Required environment variable '{var_name}' is empty after stripping comment.")
         else: # Not required, no default, empty string -> return None or empty type equivalent
              if var_type == list: return []
              if var_type == bool: return False
              if var_type in [int, float]: return var_type(0)
              return var_type() # e.g., str() -> ''

    value_to_cast = value_cleaned # Use the cleaned value

    # --- Perform Type Casting with Specific Error Handling ---
    try:
        if var_type == bool:
            result = value_to_cast.lower() in ('true', '1', 't', 'y', 'yes')
        elif var_type == list:
            try:
                # Use ast.literal_eval for safe evaluation of list-like strings
                print(f"DEBUG: Attempting ast.literal_eval on '{value_to_cast}' for '{var_name}'")
                result = ast.literal_eval(value_to_cast)
                if not isinstance(result, list):
                    # This case should ideally be caught by literal_eval raising error, but check just in case
                    raise ValueError("Value did not evaluate to a list")
            except (ValueError, SyntaxError, TypeError) as list_eval_err:
                 # Fallback to comma-separated ONLY if it doesn't look like a list/tuple literal
                 if not value_to_cast.startswith(('[', '(')) or not value_to_cast.endswith((']', ')')):
                      print(f"DEBUG: literal_eval failed for list '{var_name}', falling back to comma split. Error: {list_eval_err}")
                      result = [item.strip() for item in value_to_cast.split(',') if item.strip()]
                 else:
                      # If it looked like a list but failed, raise specific error
                      print(f"ERROR: Failed to parse list string for '{var_name}': '{value_to_cast}'. Details: {list_eval_err}")
                      raise ValueError(f"Invalid list format for env var '{var_name}'. Value='{value_to_cast}'") from list_eval_err
        elif var_type == int:
            print(f"DEBUG: Attempting int('{value_to_cast}') for '{var_name}'")
            result = int(value_to_cast)
        elif var_type == float:
            print(f"DEBUG: Attempting float('{value_to_cast}') for '{var_name}'")
            result = float(value_to_cast)
        else: # Default to string or other types
             result = var_type(value_to_cast)

        return result # Return the successfully casted result

    # --- More Specific Error Catching ---
    except ValueError as e:
        # Catch ValueError specifically from the type casting attempt
        print(f"ERROR: Failed to cast cleaned value '{value_to_cast}' for variable '{var_name}' to type {var_type}. Details: {e}")
        # Raise a new error with more context, reporting the value WE tried to cast
        raise ValueError(f"Invalid value format for environment variable '{var_name}'. Received cleaned value '{value_to_cast}' which cannot be converted to {var_type}.") from e
    except Exception as e_gen: # Catch other potential errors
         print(f"Unexpected error processing env var '{var_name}' after cleaning: {e_gen}")
         raise

# --- Logging ---
try:
    log_level_config = get_env_variable("LOG_LEVEL", "INFO").upper()
    effective_log_level = getattr(logging, log_level_config, logging.INFO)
    LOG_LEVEL = log_level_config
except ValueError as e:
     print(f"Error processing LOG_LEVEL, using INFO default: {e}")
     LOG_LEVEL = "INFO"


# --- Database Configuration ---
DATABASE_URL = get_env_variable("DATABASE_URL", required=True)

# --- API Keys ---
BINANCE_API_KEY = get_env_variable("BINANCE_API_KEY", required=True)
BINANCE_SECRET_KEY = get_env_variable("BINANCE_SECRET_KEY", required=True)
BINANCE_TLD = get_env_variable("BINANCE_TLD", "us")

CRYPTONEWS_API_TOKEN = get_env_variable("CRYPTONEWS_API_TOKEN", required=True)

REDDIT_CLIENT_ID = get_env_variable("REDDIT_CLIENT_ID", required=True)
REDDIT_CLIENT_SECRET = get_env_variable("REDDIT_CLIENT_SECRET", required=True)
REDDIT_USER_AGENT = get_env_variable("REDDIT_USER_AGENT", required=True)

TWITTER_BEARER_TOKEN = get_env_variable("TWITTER_BEARER_TOKEN", required=True)

# --- Sentiment Model ---
SENTIMENT_MODEL_NAME = get_env_variable("SENTIMENT_MODEL_NAME", "ProsusAI/finbert")

# --- Scheduling Intervals (seconds) ---
PRICE_FETCH_INTERVAL = get_env_variable("PRICE_FETCH_INTERVAL", 300, var_type=int)
INDICATOR_CALC_INTERVAL = get_env_variable("INDICATOR_CALC_INTERVAL", 300, var_type=int)
NEWS_FETCH_INTERVAL = get_env_variable("NEWS_FETCH_INTERVAL", 900, var_type=int)
REDDIT_FETCH_INTERVAL = get_env_variable("REDDIT_FETCH_INTERVAL", 900, var_type=int)
TWITTER_FETCH_INTERVAL = get_env_variable("TWITTER_FETCH_INTERVAL", 900, var_type=int)
SENTIMENT_ANALYSIS_INTERVAL = get_env_variable("SENTIMENT_ANALYSIS_INTERVAL", 1800, var_type=int)
# FEATURE_GENERATION_INTERVAL = get_env_variable("FEATURE_GENERATION_INTERVAL", 300, var_type=int) # Not scheduled separately
# MODEL_PREDICTION_INTERVAL = get_env_variable("MODEL_PREDICTION_INTERVAL", 300, var_type=int) # Not scheduled separately
TRADING_LOGIC_INTERVAL = get_env_variable("TRADING_LOGIC_INTERVAL", 300, var_type=int)
MODEL_RETRAIN_INTERVAL = get_env_variable("MODEL_RETRAIN_INTERVAL", 86400, var_type=int) # Daily

# --- Trading Parameters ---
INITIAL_CAPITAL_USD = get_env_variable("INITIAL_CAPITAL_USD", 10000.0, var_type=float)
TRADE_CAPITAL_PERCENTAGE = get_env_variable("TRADE_CAPITAL_PERCENTAGE", 0.01, var_type=float) # 1% default
STOP_LOSS_PCT = get_env_variable("STOP_LOSS_PCT", 0.05, var_type=float) # 5% default
TAKE_PROFIT_PCT = get_env_variable("TAKE_PROFIT_PCT", 0.10, var_type=float) # 10% default
MAX_CONCURRENT_POSITIONS = get_env_variable("MAX_CONCURRENT_POSITIONS", 8, var_type=int)
PORTFOLIO_DRAWDOWN_PCT = get_env_variable("PORTFOLIO_DRAWDOWN_PCT", 0.15, var_type=float) # 15% default
TRADE_TARGET_SYMBOL_PRICE_USD = get_env_variable("TRADE_TARGET_SYMBOL_PRICE_USD", 1.0, var_type=float)

# --- Model Training Parameters ---
FEATURE_LAG_PERIODS = get_env_variable("FEATURE_LAG_PERIODS", 20, var_type=int)
SENTIMENT_AGG_WINDOW_SHORT = get_env_variable("SENTIMENT_AGG_WINDOW_SHORT", '1h') # Pandas offset string
SENTIMENT_AGG_WINDOW_LONG = get_env_variable("SENTIMENT_AGG_WINDOW_LONG", '24h') # Pandas offset string
PREDICTION_HORIZON_PERIODS = get_env_variable("PREDICTION_HORIZON_PERIODS", 3, var_type=int) # e.g., 3 * 5min = 15min ahead

# --- Data Collection Parameters ---
TARGET_SUBREDDITS = get_env_variable("TARGET_SUBREDDITS", ['CryptoCurrency', 'Bitcoin', 'altcoin', 'SatoshiStreetBets', 'CryptoMoonShots', 'WallStreetBetsCrypto'], var_type=list)
REDDIT_POST_LIMIT = get_env_variable("REDDIT_POST_LIMIT", 25, var_type=int) # Per subreddit per fetch cycle
TWITTER_QUERY_KEYWORDS = get_env_variable("TWITTER_QUERY_KEYWORDS", ['crypto', 'bitcoin'], var_type=list) # Base keywords, will be combined with symbols
TWITTER_MAX_RESULTS_PER_FETCH = get_env_variable("TWITTER_MAX_RESULTS_PER_FETCH", 100, var_type=int) # Max per symbol/query per cycle (max 100 for recent search)

# --- Technical Indicator Parameters ---
# Define periods for indicators (can be customized)
SMA_FAST_PERIOD = get_env_variable("SMA_FAST_PERIOD", 10, var_type=int)
SMA_SLOW_PERIOD = get_env_variable("SMA_SLOW_PERIOD", 50, var_type=int)
EMA_FAST_PERIOD = get_env_variable("EMA_FAST_PERIOD", 12, var_type=int)
EMA_SLOW_PERIOD = get_env_variable("EMA_SLOW_PERIOD", 26, var_type=int)
RSI_PERIOD = get_env_variable("RSI_PERIOD", 14, var_type=int)
MACD_FAST_PERIOD = get_env_variable("MACD_FAST_PERIOD", 12, var_type=int)
MACD_SLOW_PERIOD = get_env_variable("MACD_SLOW_PERIOD", 26, var_type=int)
MACD_SIGNAL_PERIOD = get_env_variable("MACD_SIGNAL_PERIOD", 9, var_type=int)


# --- Function to check essential config ---
# (Optional: Can add more checks here if needed during initialization)
def check_essential_config():
    """Checks if all required environment variables were loaded."""
    logger = logging.getLogger(__name__) # Get logger instance
    required_vars = [
        "DATABASE_URL", "BINANCE_API_KEY", "BINANCE_SECRET_KEY",
        "CRYPTONEWS_API_TOKEN", "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET",
        "REDDIT_USER_AGENT", "TWITTER_BEARER_TOKEN"
    ]
    missing = []
    all_vars = globals() # Get all global variables in this module
    for var_name in required_vars:
        # Check if the variable exists and is not None or empty
        if not all_vars.get(var_name):
            missing.append(var_name)

    if missing:
        # Use log if configured, otherwise print
        # Check if root logger has handlers before logging
        # Use print directly as logging might not be configured if LOG_LEVEL itself is missing/invalid
        print(f"CRITICAL: Essential configuration variables missing or empty: {', '.join(missing)}")
        print("Please set them via environment variables or a .env file.")
        return False
    # Use log if configured, otherwise print
    logger = logging.getLogger() # Get root logger
    if logger.hasHandlers() and logger.isEnabledFor(logging.INFO):
         log.info("Essential configuration variables seem present.")
    else:
         print("INFO: Essential configuration variables seem present.")
    return True

# --- Log loaded config (optional, be careful with sensitive data) ---
# Logging should be configured in main.py before this module is fully imported usually
# We use print statements within get_env_variable for early feedback during startup.

