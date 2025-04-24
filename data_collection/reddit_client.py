# /data_collection/reddit_client.py

import logging
import datetime
import pytz
from typing import List, Dict, Optional
import asyncio
import asyncpraw
from asyncpraw.models import Subreddit
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from database.db_utils import async_bulk_insert

log = logging.getLogger(__name__)

# Instead of initializing Reddit directly, create a function to get the Reddit client
# This prevents the "no event loop" error when importing the module
_reddit_client = None

async def get_reddit_client():
    """Get or create the Reddit API client"""
    global _reddit_client
    if _reddit_client is None:
        _reddit_client = asyncpraw.Reddit(
            client_id=config.REDDIT_CLIENT_ID,
            client_secret=config.REDDIT_CLIENT_SECRET,
            user_agent=config.REDDIT_USER_AGENT
        )
    return _reddit_client

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_new_subreddit_posts(subreddit_names: List[str], post_limit_per_subreddit: int = config.REDDIT_POST_LIMIT) -> List[Dict]:
    """
    Fetch new posts from specified subreddits.
    Returns list of processed posts.
    """
    log.info(f"Fetching up to {post_limit_per_subreddit} new posts from subreddits: {subreddit_names}")
    
    all_posts = []
    reddit = await get_reddit_client()

    for sub_name in subreddit_names:
        try:
            log.debug(f"Accessing subreddit: r/{sub_name}")
            subreddit = await reddit.subreddit(sub_name)
            async for post in subreddit.new(limit=post_limit_per_subreddit):
                if post.stickied:
                    log.debug(f"Skipping stickied post in r/{sub_name}: {post.id}")
                    continue

                post_text_combined = post.title
                if post.is_self and post.selftext:
                    post_text_combined += "\n" + post.selftext

                created_utc_dt = datetime.datetime.fromtimestamp(post.created_utc, tz=pytz.utc)

                processed_post = {
                    'post_id': post.id,
                    'subreddit': post.subreddit.display_name,
                    'title': post.title,
                    'selftext': post.selftext if post.is_self else None,
                    'url': post.url,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': created_utc_dt,
                    'fetched_at': datetime.datetime.now(pytz.utc)
                }
                all_posts.append(processed_post)

            log.debug(f"Fetched {len(all_posts)} posts from r/{sub_name}")
            
            # Small delay between subreddits
            await asyncio.sleep(0.5)

        except Exception as e:
            log.error(f"Error fetching posts from r/{sub_name}: {e}", exc_info=True)
            continue

    # Store posts in database
    if all_posts:
        try:
            await async_bulk_insert(all_posts, 'reddit_data')
            log.info(f"Successfully stored {len(all_posts)} Reddit posts")
        except Exception as e:
            log.error(f"Error storing Reddit posts in database: {e}")

    return all_posts

async def fetch_reddit_posts(
    query: str,
    after: Optional[datetime.datetime] = None,
    limit: int = 50
) -> List[Dict]:
    """
    Fetch Reddit posts for a specific query for the dashboard.
    
    Args:
        query: Search query for subreddits or keywords
        after: Only include posts after this timestamp
        limit: Maximum number of posts to fetch
        
    Returns:
        List of processed Reddit posts
    """
    try:
        log.info(f"Fetching Reddit posts for query: {query}")
        
        # For the dashboard, we'll use the query to determine which subreddits to check
        crypto_subreddits = config.REDDIT_SUBREDDITS
        
        # Get posts from relevant subreddits
        posts = await fetch_new_subreddit_posts(crypto_subreddits, limit)
        
        # Filter posts that match the query
        if query:
            query_lower = query.lower()
            filtered_posts = []
            
            for post in posts:
                # Check if query matches in title or selftext
                title = post.get('title', '').lower()
                selftext = post.get('selftext', '') or ''
                selftext = selftext.lower()
                
                if query_lower in title or query_lower in selftext:
                    filtered_posts.append(post)
            
            posts = filtered_posts
        
        # Filter by date if needed
        if after:
            posts = [post for post in posts if post['created_utc'] >= after]
            
        log.info(f"Found {len(posts)} Reddit posts matching query: {query}")
        return posts
        
    except Exception as e:
        log.error(f"Error fetching Reddit posts for query {query}: {e}")
        return []

