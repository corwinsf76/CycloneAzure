# /sentiment_analysis/analyzer.py

import logging
from typing import List, Dict, Optional, Tuple, Any, Union, cast
# Fix imports to avoid Pylance errors by using fully qualified paths
from transformers.pipelines import pipeline
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
# Import PreTrainedTokenizerBase for better typing
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import datetime
import pytz
import time
import torch
import pandas as pd
from sqlalchemy import text

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config
from database import db_utils
# from .. import config # Use relative import if running as part of a package

log = logging.getLogger(__name__)

# --- Model Initialization ---
_sentiment_pipeline = None
_tokenizer = None
_model = None

def load_sentiment_model() -> Optional[Tuple[Any, PreTrainedTokenizerBase]]:
    """
    Loads the Hugging Face sentiment analysis pipeline specified in the config.
    Uses a singleton pattern to load the model only once.

    Returns:
        Optional[Tuple[Any, PreTrainedTokenizerBase]]: The loaded pipeline and tokenizer,
                                            or None if loading fails.
    """
    global _sentiment_pipeline, _tokenizer, _model
    if _sentiment_pipeline is None:
        model_name = config.SENTIMENT_MODEL_NAME
        log.info(f"Attempting to load sentiment analysis model: {model_name}")
        start_time = time.time()
        try:
            # Load tokenizer and model explicitly for more control if needed
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # Use GPU if available, otherwise fallback to CPU
            device = 0 if torch.cuda.is_available() else -1

            # Create the pipeline but use Any type to avoid Pylance errors
            _sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=_model,
                tokenizer=_tokenizer,
                device=device,  # Specify device for acceleration
            )
            end_time = time.time()
            log.info(f"Sentiment analysis model '{model_name}' loaded successfully in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            log.error(f"Error loading sentiment model '{model_name}': {e}", exc_info=True)
            _sentiment_pipeline = None
            _tokenizer = None
            _model = None

    if _sentiment_pipeline and _tokenizer:
         return _sentiment_pipeline, _tokenizer
    else:
         return None


def analyze_sentiment_batch(items_with_text: List[Dict], text_key: str = 'text', batch_size: int = 64) -> List[Dict]:
    """
    Analyzes sentiment for text within a list of dictionaries using the loaded pipeline.

    Args:
        items_with_text (List[Dict]): A list of dictionaries, where each dictionary
                                      must contain at least the key specified by `text_key`.
                                      It should also contain identifier keys (like 'id', 'source_type')
                                      to link the results back.
        text_key (str): The key in each dictionary that holds the text to be analyzed.
                        Defaults to 'text'.
        batch_size (int): Number of items to process in each batch. Defaults to 64.

    Returns:
        List[Dict]: A list of dictionaries, each containing:
                    - 'source_id': The original 'id' from the input item.
                    - 'sentiment_label': 'positive', 'negative', or 'neutral' (lowercase).
                    - 'sentiment_score': Float score (-1.0 to 1.0, positive for positive, negative for negative).
                    - 'analyzed_at': Timestamp of analysis (timezone-aware UTC datetime).
                    Returns an empty list if analysis fails or no valid text is found.
    """
    model_info = load_sentiment_model()
    if not model_info:
        log.error("Sentiment pipeline not available. Cannot analyze sentiment.")
        return []

    pipeline_obj, tokenizer = model_info  # Renamed to avoid shadowing the imported pipeline function

    if not items_with_text:
        log.warning("No items provided for sentiment analysis.")
        return []

    results = []
    texts_to_analyze = []
    original_indices = [] # To map results back to original items

    # Prepare batch of texts and keep track of original item IDs
    log.info(f"Preparing {len(items_with_text)} items for sentiment analysis...")
    for i, item in enumerate(items_with_text):
        text = item.get(text_key)
        item_id = item.get('id') # Assuming 'id' is the primary key from the source table

        if item_id is None:
            log.warning(f"Skipping item at index {i} due to missing 'id'. Item keys: {item.keys()}")
            continue

        # Basic text cleaning (optional, can be expanded)
        if isinstance(text, str) and text.strip():
            cleaned_text = text.strip().replace('\n', ' ').replace('\r', '') # Remove newlines for analysis
            texts_to_analyze.append(cleaned_text)
            original_indices.append(i)
        else:
            log.debug(f"Skipping item id={item_id} due to missing or invalid text.")

    if not texts_to_analyze:
        log.warning("No valid text found in items for sentiment analysis.")
        return []

    log.info(f"Analyzing sentiment for {len(texts_to_analyze)} text items in batches of {batch_size}...")
    analysis_timestamp = datetime.datetime.now(pytz.utc)
    analysis_results = []

    try:
        for i in range(0, len(texts_to_analyze), batch_size):
            batch_texts = texts_to_analyze[i:i + batch_size]
            # Set max_length with better type safety for tokenizers
            max_length = 512  # Safe default value
            
            # Safe type-checked tokenizer attribute access
            try:
                # First try direct attribute access with type checking
                if hasattr(tokenizer, "model_max_length"):
                    model_max_attr = getattr(tokenizer, "model_max_length")
                    if isinstance(model_max_attr, int) and model_max_attr > 0:
                        max_length = model_max_attr
                # Try accessing through tokenizer.init_kwargs if available
                elif hasattr(tokenizer, "init_kwargs") and isinstance(tokenizer.init_kwargs, dict):
                    init_kwargs = tokenizer.init_kwargs
                    if "model_max_length" in init_kwargs and isinstance(init_kwargs["model_max_length"], int):
                        max_length = init_kwargs["model_max_length"]
                # Last resort: check if 'max_len' exists in __dict__
                elif hasattr(tokenizer, "__dict__") and "max_len" in tokenizer.__dict__:
                    if isinstance(tokenizer.__dict__["max_len"], int):
                        max_length = tokenizer.__dict__["max_len"]
            except (AttributeError, TypeError, ValueError) as e:
                log.warning(f"Could not determine tokenizer max_length: {e}, using default 512")
            
            # Ensure max_length is a proper integer
            max_length = int(max_length)
            log.debug(f"Using max_length={max_length} for tokenization")
            
            batch_results = pipeline_obj(batch_texts, truncation=True, max_length=max_length)
            analysis_results.extend(batch_results)

        if len(analysis_results) != len(original_indices):
             log.error(f"Mismatch in sentiment output count ({len(analysis_results)}) vs input text count ({len(texts_to_analyze)}).")
             return [] # Avoid partial results or incorrect mapping

        # Process results and map back
        for i, sentiment_output in enumerate(analysis_results):
            original_item_index = original_indices[i]
            original_item = items_with_text[original_item_index]
            original_item_id = original_item['id'] # We ensured ID exists earlier

            # Add type safety for accessing dictionary keys
            label = sentiment_output.get('label', 'neutral').lower()
            score = float(sentiment_output.get('score', 0.0))
            normalized_score = 0.0

            if label == 'positive':
                normalized_score = score
            elif label == 'negative':
                normalized_score = -score

            results.append({
                'source_id': original_item_id, # Link back to the original record ID
                'sentiment_label': label,
                'sentiment_score': round(normalized_score, 4),
                'analyzed_at': analysis_timestamp
            })

    except Exception as e:
        log.error(f"Error during sentiment analysis pipeline execution: {e}", exc_info=True)

    log.info(f"Successfully analyzed sentiment for {len(results)} items.")
    return results

def get_current_sentiment_score(symbol: str, window: str = '24h') -> float:
    """
    Retrieves the current sentiment score for a symbol from the database.
    
    Args:
        symbol: The cryptocurrency symbol to get sentiment for (e.g., 'BTC')
        window: Time window to aggregate sentiment over ('1h', '24h', '7d')
        
    Returns:
        float: Sentiment score from -1.0 (very negative) to 1.0 (very positive)
              Returns 0.0 if no data is available
    """
    try:
        # Default to 0.0 for neutral sentiment if no data is available
        score = 0.0
        
        # Calculate time window
        now = datetime.datetime.now(pytz.utc)
        if window == '1h':
            start_time = now - datetime.timedelta(hours=1)
        elif window == '24h':
            start_time = now - datetime.timedelta(days=1)
        elif window == '7d':
            start_time = now - datetime.timedelta(days=7)
        else:
            # Default to 24 hours
            start_time = now - datetime.timedelta(days=1)
            log.warning(f"Unknown window '{window}', defaulting to 24h")
        
        # Connect to database
        if not db_utils.engine:
            log.error("Database engine not available")
            return score
            
        with db_utils.engine.connect() as connection:
            # Query for sentiment data
            # First try CryptoPanic sentiment data
            query = text("""
                SELECT AVG(sentiment_score) as avg_score 
                FROM cryptopanic_sentiment 
                WHERE symbol = :symbol 
                AND timestamp >= :start_time
            """)
            
            result = connection.execute(query, {"symbol": symbol, "start_time": start_time})
            row = result.fetchone()
            
            if row and row[0] is not None:
                score = float(row[0])
                log.debug(f"Found CryptoPanic sentiment for {symbol}: {score}")
                return score
                
            # Fallback to sentiment_analysis_results if no CryptoPanic data
            # This assumes sentiment_analysis_results has been populated and has a way to filter by symbol
            # You may need to adjust this query based on your actual schema
            query = text("""
                SELECT AVG(sentiment_score) as avg_score 
                FROM sentiment_analysis_results 
                WHERE analyzed_at >= :start_time
                -- Add your logic to filter by symbol
            """)
            
            result = connection.execute(query, {"start_time": start_time})
            row = result.fetchone()
            
            if row and row[0] is not None:
                score = float(row[0])
                log.debug(f"Found sentiment analysis results: {score}")
                
        return score
    except Exception as e:
        log.error(f"Error getting sentiment score for {symbol}: {e}")
        return 0.0  # Return neutral sentiment on error

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # Setup basic logging for direct script run
    logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s [%(name)s] %(message)s')

    print("--- Testing Sentiment Analyzer ---")

    # Ensure model is loaded (call function directly)
    load_sentiment_model()

    # Sample data mimicking fetched data (needs 'id' and a text key)
    sample_data = [
        {'id': 101, 'source_type': 'reddit', 'text': "This is great news, Bitcoin is going to the moon!🚀"},
        {'id': 102, 'source_type': 'twitter', 'text': "Lost all my savings in $XYZ coin, feeling devastated. Total scam."},
        {'id': 103, 'source_type': 'news', 'text': "Market analysts suggest a period of consolidation for major cryptocurrencies."},
        {'id': 104, 'source_type': 'reddit', 'text': ""}, # Empty text
        {'id': 105, 'source_type': 'twitter'}, # Missing text key
        {'id': 106, 'source_type': 'news', 'text': "Neutral outlook for the short term."},
    ]

    print(f"\nAnalyzing {len(sample_data)} sample items...")
    # Specify the key containing the text, e.g., 'text' or 'text_combined'
    sentiment_results = analyze_sentiment_batch(sample_data, text_key='text')

    if sentiment_results:
        print(f"\nAnalysis Results ({len(sentiment_results)}):")
        for res in sentiment_results:
            print(f"  Source ID: {res['source_id']}, Label: {res['sentiment_label']}, Score: {res['sentiment_score']:.4f}")
    else:
        print("\nSentiment analysis failed or produced no results. Check logs (model loading?).")

    print("\n--- Test Complete ---")

