# /config.py
import os
import logging
from dotenv import load_dotenv
import sys  # Import sys for checking handlers
import ast  # Import ast for literal_eval
from typing import TypeVar, Optional, Union, List, Type, Any, cast, overload

# Create specific type variables for each supported type
StrT = TypeVar('StrT', bound=str)
IntT = TypeVar('IntT', bound=int)
FloatT = TypeVar('FloatT', bound=float)
BoolT = TypeVar('BoolT', bound=bool)
ListT = TypeVar('ListT', bound=list)

# --- Load .env File ---
# Load environment variables from .env file if it exists
# Searches current directory and parents.
env_path = load_dotenv()
if env_path:
    # Use print before logging is configured
    print(f"Loaded environment variables from: {env_path}")
else:
    # Use print before logging is configured
    print("INFO: .env file not found or empty. Relying on system environment variables.")

# Configure logging temporarily here if needed for early config errors
# logging.basicConfig(level=logging.DEBUG) # Example temporary config
log = logging.getLogger(__name__)

@overload
def get_env_variable(var_name: str, default: Optional[str] = None, required: bool = False, var_type: Type[str] = str) -> str: ...

@overload
def get_env_variable(var_name: str, default: Optional[int] = None, required: bool = False, var_type: Type[int] = int) -> int: ...

@overload
def get_env_variable(var_name: str, default: Optional[float] = None, required: bool = False, var_type: Type[float] = float) -> float: ...

@overload
def get_env_variable(var_name: str, default: Optional[bool] = None, required: bool = False, var_type: Type[bool] = bool) -> bool: ...

@overload
def get_env_variable(var_name: str, default: Optional[List[Any]] = None, required: bool = False, var_type: Type[List[Any]] = list) -> List[Any]: ...

def get_env_variable(
    var_name: str,
    default: Any = None,
    required: bool = False,
    var_type: Type[Any] = str
) -> Any:
    """
    Retrieves an environment variable with type casting and validation.
    
    Args:
        var_name: Name of the environment variable
        default: Default value if variable is not set
        required: Whether the variable is required
        var_type: Type to cast the value to
    
    Returns:
        The environment variable value cast to the specified type
    
    Raises:
        ValueError: If required variable is not set or type casting fails
    """
    raw_value = os.getenv(var_name)

    if raw_value is None:
        if required:
            print(f"CRITICAL: Required environment variable '{var_name}' is not set.")
            raise ValueError(f"Required environment variable '{var_name}' is not set.")
        return default if default is not None else cast(Any, var_type())

    value_cleaned = raw_value
    if isinstance(raw_value, str):
        value_cleaned = raw_value.split('#')[0].strip()

    if not value_cleaned:
        if default is not None:
            return default
        elif required:
            raise ValueError(f"Required environment variable '{var_name}' is empty after stripping comment.")
        else:
            if var_type == list:
                return cast(List[Any], [])
            if var_type == bool:
                return cast(bool, False)
            if var_type in (int, float):
                return cast(Any, var_type(0))
            return cast(Any, var_type())

    try:
        value_to_cast = value_cleaned

        if var_type == bool:
            return cast(bool, value_to_cast.lower() in ('true', '1', 'yes', 'on'))
        elif var_type == list:
            try:
                result = ast.literal_eval(value_to_cast)
                if not isinstance(result, list):
                    raise ValueError(f"Value '{value_to_cast}' did not evaluate to a list")
                return cast(List[Any], result)
            except (SyntaxError, ValueError):
                return cast(List[str], [item.strip() for item in value_to_cast.split(',') if item.strip()])
        elif var_type == int:
            return cast(int, int(value_to_cast))
        elif var_type == float:
            return cast(float, float(value_to_cast))
        elif var_type == str:
            return cast(str, value_to_cast)
        else:
            return cast(Any, var_type(value_to_cast))

    except ValueError as e:
        print(f"ERROR: Failed to cast cleaned value '{value_to_cast}' for variable '{var_name}' to type {var_type}. Details: {e}")
        raise ValueError(f"Invalid value format for environment variable '{var_name}'. Received cleaned value '{value_to_cast}' which cannot be converted to {var_type}.") from e
    except Exception as e_gen:
        print(f"Unexpected error processing env var '{var_name}' after cleaning: {e_gen}")
        raise

def get_env_variable_flexible(
    var_names: Union[str, List[str]],
    default: Any = None,
    required: bool = False,
    var_type: Type[Any] = str
) -> Any:
    """
    Retrieves an environment variable with flexible naming support.
    Tries multiple possible names for the same variable.
    
    Args:
        var_names: Name or list of names for the environment variable
        default: Default value if variable is not set
        required: Whether the variable is required
        var_type: Type to cast the value to
    
    Returns:
        The environment variable value cast to the specified type
    
    Raises:
        ValueError: If required variable is not set or type casting fails
    """
    # Convert single name to list
    names_to_try = [var_names] if isinstance(var_names, str) else var_names
    
    # Try each name
    for name in names_to_try:
        raw_value = os.getenv(name)
        if raw_value is not None:
            # Process the found value using similar logic to get_env_variable
            value_cleaned = raw_value
            if isinstance(raw_value, str):
                value_cleaned = raw_value.split('#')[0].strip()
                
            if not value_cleaned:
                continue  # Try next name if this value is empty after cleaning
                
            try:
                value_to_cast = value_cleaned
                
                if var_type == bool:
                    return value_to_cast.lower() in ('true', '1', 'yes', 'on')
                elif var_type == list:
                    try:
                        result = ast.literal_eval(value_to_cast)
                        if not isinstance(result, list):
                            raise ValueError(f"Value '{value_to_cast}' did not evaluate to a list")
                        return result
                    except (SyntaxError, ValueError):
                        return [item.strip() for item in value_to_cast.split(',') if item.strip()]
                elif var_type == int:
                    return int(value_to_cast)
                elif var_type == float:
                    return float(value_to_cast)
                elif var_type == str:
                    return value_to_cast
                else:
                    return var_type(value_to_cast)
                    
            except Exception as e:
                print(f"Warning: Failed to process value '{value_to_cast}' for variable '{name}': {e}")
                continue  # Try next name
    
    # If we're here, no valid value was found with any name
    if required:
        tried_names_str = ", ".join(names_to_try)
        print(f"CRITICAL: Required environment variable (tried: {tried_names_str}) is not set.")
        raise ValueError(f"Required environment variable (tried: {tried_names_str}) is not set.")
        
    # Return default if not required and no valid value found
    return default if default is not None else cast(Any, var_type())

# --- Logging ---
try:
    log_level_config = get_env_variable("LOG_LEVEL", "INFO", var_type=str).upper()
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

# API Settings for new integrations
CRYPTOPANIC_API_TOKEN = get_env_variable_flexible(
    ['CRYPTOPANIC_API_TOKEN', 'CryptoPanic_API_Token', 'CRYPTOPANIC_API_KEY', 'CryptoPanic_API_Key'],
    default='',
    required=False
)

# AlphaVantage Configuration - use flexible naming convention to find the key
ALPHAVANTAGE_API_KEY = get_env_variable_flexible(
    ['ALPHAVANTAGE_API_KEY', 'AlphaVantage_API_Key', 'ALPHAVANTAGE_API_TOKEN', 'AlphaVantage_API_Token'],
    default='dummy_key_for_testing',  # Add default for testing
    required=False  # Changed from True to False to allow testing without this key
)
ALPHAVANTAGE_BASE_URL = 'https://www.alphavantage.co/query'
ALPHAVANTAGE_RATE_LIMIT = 5  # Calls per minute for free tier
ALPHAVANTAGE_RATE_LIMIT_PERIOD = 60  # Period in seconds

# CoinGecko Configuration
COINGECKO_API_KEY = get_env_variable_flexible(
    ['COINGECKO_API_KEY', 'CoinGecko_API_Key', 'COINGECKO_API_TOKEN', 'CoinGecko_API_Token'],
    default=''  # Make it optional as CoinGecko has free tier without key
)
COINGECKO_CALLS_PER_MINUTE = 50
COINGECKO_FETCH_INTERVAL = 300  # 5 minutes

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
MODEL_RETRAIN_INTERVAL = get_env_variable("MODEL_RETRAIN_INTERVAL", 86400, var_type=int)  # Daily

# --- API Rate Limits and Intervals ---
# CryptoCompare (free tier: 100k calls/month)
CRYPTOCOMPARE_CALLS_PER_MINUTE = 30
CRYPTOCOMPARE_FETCH_INTERVAL = 300  # 5 minutes

# AlphaVantage (free tier: 5 calls/minute, 500 calls/day)
ALPHAVANTAGE_CALLS_PER_MINUTE = 5
ALPHAVANTAGE_FETCH_INTERVAL = 900  # 15 minutes

# CoinGecko (free tier: 10-50 calls/minute)
COINGECKO_CALLS_PER_MINUTE = 10
COINGECKO_FETCH_INTERVAL = 300  # 5 minutes

# CryptoPanic (free tier limits)
CRYPTOPANIC_CALLS_PER_MINUTE = 15
CRYPTOPANIC_FETCH_INTERVAL = 600  # 10 minutes

# Santiment (free tier limits)
SANTIMENT_CALLS_PER_MINUTE = 10
SANTIMENT_FETCH_INTERVAL = 900  # 15 minutes

# Data backfill settings
BACKFILL_DAYS = 30  # Number of days to backfill by default

# --- Trading Parameters ---
INITIAL_CAPITAL_USD = get_env_variable("INITIAL_CAPITAL_USD", 10000.0, var_type=float)
TRADE_CAPITAL_PERCENTAGE = 0.02  # Increased from 1% to 2% for higher returns
STOP_LOSS_PCT = 0.03  # Reduced from 5% to minimize losses
TAKE_PROFIT_PCT = 0.15  # Increased from 10% to capture larger profits
MAX_CONCURRENT_POSITIONS = 12  # Increased from 8 to allow more trades
PORTFOLIO_DRAWDOWN_PCT = get_env_variable("PORTFOLIO_DRAWDOWN_PCT", 0.15, var_type=float)  # 15% default
TRADE_TARGET_SYMBOL_PRICE_USD = get_env_variable("TRADE_TARGET_SYMBOL_PRICE_USD", 1.0, var_type=float)
BUY_CONFIDENCE_THRESHOLD = 0.55  # Lowered from 0.65 to increase trade frequency
SELL_CONFIDENCE_THRESHOLD = 0.55  # Lowered from 0.65 to align with buy threshold

# --- Model Training Parameters ---
FEATURE_LAG_PERIODS = get_env_variable("FEATURE_LAG_PERIODS", 20, var_type=int)
SENTIMENT_AGG_WINDOW_SHORT = get_env_variable("SENTIMENT_AGG_WINDOW_SHORT", '1h')  # Pandas offset string
SENTIMENT_AGG_WINDOW_LONG = get_env_variable("SENTIMENT_AGG_WINDOW_LONG", '24h')  # Pandas offset string
PREDICTION_HORIZON_PERIODS = get_env_variable("PREDICTION_HORIZON_PERIODS", 3, var_type=int)  # e.g., 3 * 5min = 15min ahead

# --- Data Collection Parameters ---
CANDLE_INTERVAL = get_env_variable("CANDLE_INTERVAL", "5m", var_type=str)  # Default to 5-minute candles
TARGET_SUBREDDITS = get_env_variable("TARGET_SUBREDDITS", ['CryptoCurrency', 'Bitcoin', 'altcoin', 'SatoshiStreetBets', 'CryptoMoonShots', 'WallStreetBetsCrypto'], var_type=list)
REDDIT_POST_LIMIT = get_env_variable("REDDIT_POST_LIMIT", 25, var_type=int)  # Per subreddit per fetch cycle
TWITTER_QUERY_KEYWORDS = get_env_variable("TWITTER_QUERY_KEYWORDS", ['crypto', 'bitcoin'], var_type=list)  # Base keywords, will be combined with symbols
TWITTER_MAX_RESULTS_PER_FETCH = get_env_variable("TWITTER_MAX_RESULTS_PER_FETCH", 100, var_type=int)  # Max per symbol/query per cycle (max 100 for recent search)

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

# Base cryptocurrency symbols to track
BASE_SYMBOLS: List[str] = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'DOT']

# --- Function to check essential config ---
def check_essential_config():
    """Checks if all required environment variables were loaded."""
    required_vars = [
        "DATABASE_URL", "BINANCE_API_KEY", "BINANCE_SECRET_KEY",
        "CRYPTONEWS_API_TOKEN", "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET",
        "REDDIT_USER_AGENT", "TWITTER_BEARER_TOKEN"
    ]
    missing = [var for var in required_vars if not globals().get(var)]
    if missing:
        print(f"CRITICAL: Missing essential config variables: {', '.join(missing)}")
        return False
    return True

# --- Log loaded config (optional, be careful with sensitive data) ---
# Logging should be configured in main.py before this module is fully imported usually
# We use print statements within get_env_variable for early feedback during startup.

