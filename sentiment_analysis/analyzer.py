"""
Sentiment Analysis Module - Now with async support

This module handles sentiment analysis of text data using FinBERT only.
Optimized for PostgreSQL storage and async operation.
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os

import config
from database.db_utils import async_bulk_insert

log = logging.getLogger(__name__)

# Global variables for the sentiment model
_sentiment_model = None
_sentiment_tokenizer = None

# Initialize thread pool for CPU-bound text processing
_thread_pool = ThreadPoolExecutor(max_workers=4)

def load_sentiment_model() -> Any:
    """
    Loads the pre-trained sentiment analysis model.
    Uses a singleton pattern to load only once.
    
    Returns:
        The model object or None if loading fails.
    """
    global _sentiment_model, _sentiment_tokenizer
    
    if _sentiment_model is not None:
        return _sentiment_model
    
    try:
        log.info(f"Loading sentiment model: {config.SENTIMENT_MODEL_NAME}")
        
        # Import here to avoid loading transformers at module initialization
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        # Load the model and tokenizer
        _sentiment_tokenizer = AutoTokenizer.from_pretrained(config.SENTIMENT_MODEL_NAME)
        _sentiment_model = AutoModelForSequenceClassification.from_pretrained(config.SENTIMENT_MODEL_NAME)
        
        log.info(f"Sentiment model loaded successfully")
        return _sentiment_model
    except Exception as e:
        log.error(f"Failed to load sentiment model: {e}", exc_info=True)
        return None

def _analyze_text_with_finbert(text: str) -> Dict[str, float]:
    """
    Analyzes text using the FinBERT model specifically trained for financial text.
    Only FinBERT is used; no fallback to TextBlob or neutral.
    """
    global _sentiment_model, _sentiment_tokenizer
    
    # Ensure model is loaded
    if _sentiment_model is None or _sentiment_tokenizer is None:
        if load_sentiment_model() is None:
            raise RuntimeError("FinBERT model could not be loaded. Sentiment analysis cannot proceed.")
    try:
        clean_text = ' '.join(text.split())
        max_length = 512
        if len(clean_text.split()) > max_length:
            clean_text = ' '.join(clean_text.split()[:max_length])
        import torch
        inputs = _sentiment_tokenizer(clean_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = _sentiment_model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        negative_score = float(scores[0][0])
        neutral_score = float(scores[0][1])
        positive_score = float(scores[0][2])
        sentiment_score = positive_score - negative_score
        magnitude = abs(sentiment_score) + (1 - neutral_score)
        subjectivity = 1 - neutral_score
        return {
            'score': sentiment_score,
            'magnitude': magnitude,
            'subjectivity': subjectivity,
            'positive': positive_score,
            'negative': negative_score,
            'neutral': neutral_score
        }
    except Exception as e:
        log.error(f"Error in FinBERT analysis: {e}", exc_info=True)
        raise

async def analyze_sentiment(text: str) -> Optional[Dict[str, float]]:
    """
    Analyze sentiment of text asynchronously using FinBERT only.
    Returns a dictionary with sentiment metrics.
    """
    if not text or len(text.strip()) == 0:
        return None
    try:
        loop = asyncio.get_running_loop()
        sentiment = await loop.run_in_executor(
            _thread_pool,
            _analyze_text_with_finbert,
            text
        )
        result = {
            'text': text[:500],
            'sentiment_score': sentiment['score'],
            'sentiment_magnitude': sentiment['magnitude'],
            'subjectivity': sentiment['subjectivity'],
            'positive_score': sentiment.get('positive', None),
            'negative_score': sentiment.get('negative', None),
            'neutral_score': sentiment.get('neutral', None),
            'model_used': 'finbert'
        }
        await async_bulk_insert([result], 'sentiment_analysis_results')
        return sentiment
    except Exception as e:
        log.error(f"Error in sentiment analysis: {e}")
        return None

async def analyze_texts_batch(texts: List[str], batch_size: int = 100) -> List[Dict[str, float]]:
    """
    Analyze sentiment for a batch of texts concurrently.
    Returns list of sentiment results.
    """
    if not texts:
        return []
    
    try:
        # Process texts in batches
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Create tasks for batch
            tasks = [analyze_sentiment(text) for text in batch]
            
            # Run batch concurrently
            batch_results = await asyncio.gather(*tasks)
            results.extend([r for r in batch_results if r is not None])
            
            # Small delay between batches
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return results
        
    except Exception as e:
        log.error(f"Error in batch sentiment analysis: {e}")
        return []

async def get_aggregate_sentiment(texts: List[str]) -> Optional[Dict[str, float]]:
    """
    Calculate aggregate sentiment metrics for a list of texts.
    Returns dictionary with aggregated metrics.
    """
    if not texts:
        return None
    
    try:
        # Analyze all texts
        sentiments = await analyze_texts_batch(texts)
        
        if not sentiments:
            return None
        
        # Calculate aggregate metrics
        scores = [s['score'] for s in sentiments]
        magnitudes = [s['magnitude'] for s in sentiments]
        
        aggregate = {
            'avg_score': float(np.mean(scores)),
            'avg_magnitude': float(np.mean(magnitudes)),
            'std_score': float(np.std(scores)),
            'max_score': float(np.max(scores)),
            'min_score': float(np.min(scores)),
            'sample_size': len(sentiments)
        }
        
        # Store aggregate results
        await async_bulk_insert([aggregate], 'sentiment_aggregates')
        return aggregate
        
    except Exception as e:
        log.error(f"Error calculating aggregate sentiment: {e}")
        return None

