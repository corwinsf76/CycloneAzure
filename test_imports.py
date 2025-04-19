"""Test script to validate package imports."""
try:
    import pandas as pd
    import numpy as np
    import dash
    import dash_bootstrap_components as dbc
    from dash import html
    import plotly.graph_objects as go
    from sqlalchemy import create_engine
    import schedule
    import lightgbm
    # Fix problematic transformer imports with fully qualified path names
    from transformers.pipelines import pipeline
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
    import praw
    import tweepy
    import pytz
    from binance.client import Client
    print("All core package imports successful!")
except ImportError as e:
    print(f"Import error: {e}")