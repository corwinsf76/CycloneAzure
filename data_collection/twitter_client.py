# data_collection/twitter_client.py

import os
import tweepy

# Load Twitter credentials
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Initialize Tweepy Client
client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

def search_recent_tweets(query, max_results=100):
    """
    Search recent tweets using Twitter v2 API (official) with Tweepy.
    """
    response = client.search_recent_tweets(
        query=f"({query}) lang:en -is:retweet",
        max_results=max_results,
        tweet_fields=["created_at", "public_metrics", "author_id", "lang"]
    )

    if response.data is None:
        return []

    return response.data
