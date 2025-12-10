"""
News fetching utilities for VolRegime frontend
Handles financial news from RSS feeds and other sources
"""

import feedparser
import streamlit as st
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
from urllib.parse import urlparse

@st.cache_data(ttl=900)  # Cache for 15 minutes
def get_yahoo_finance_news(limit: int = 10) -> List[Dict]:
    """
    Fetch latest financial news from Yahoo Finance RSS feed
    
    Args:
        limit: Maximum number of articles to return
    
    Returns:
        List of news articles with title, summary, link, and published date
    """
    try:
        # Yahoo Finance RSS feed URL
        rss_url = "https://feeds.finance.yahoo.com/rss/2.0/headline"
        
        # Parse the RSS feed
        feed = feedparser.parse(rss_url)
        
        articles = []
        for entry in feed.entries[:limit]:
            # Parse published date
            try:
                published = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %z")
                published_str = published.strftime("%Y-%m-%d %H:%M")
            except:
                published_str = entry.get('published', 'Unknown date')
            
            article = {
                'title': entry.title,
                'summary': entry.get('summary', entry.get('description', 'No summary available')),
                'link': entry.link,
                'published': published_str,
                'source': 'Yahoo Finance',
                'category': 'General Finance'
            }
            articles.append(article)
        
        return articles
        
    except Exception as e:
        st.warning(f"Error fetching Yahoo Finance news: {str(e)}")
        return get_fallback_news()

@st.cache_data(ttl=900)
def get_marketwatch_news(limit: int = 10) -> List[Dict]:
    """
    Fetch financial news from MarketWatch RSS feed
    
    Args:
        limit: Maximum number of articles to return
    
    Returns:
        List of news articles
    """
    try:
        rss_url = "http://feeds.marketwatch.com/marketwatch/topstories/"
        feed = feedparser.parse(rss_url)
        
        articles = []
        for entry in feed.entries[:limit]:
            try:
                published = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %z")
                published_str = published.strftime("%Y-%m-%d %H:%M")
            except:
                published_str = entry.get('published', 'Unknown date')
            
            article = {
                'title': entry.title,
                'summary': entry.get('summary', entry.get('description', 'No summary available')),
                'link': entry.link,
                'published': published_str,
                'source': 'MarketWatch',
                'category': 'Market News'
            }
            articles.append(article)
        
        return articles
        
    except Exception as e:
        st.warning(f"Error fetching MarketWatch news: {str(e)}")
        return []

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_custom_articles() -> List[Dict]:
    """
    Load custom articles from local JSON file
    
    Returns:
        List of custom articles
    """
    try:
        # Path to custom articles JSON file
        articles_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'custom_articles.json')
        
        if os.path.exists(articles_file):
            with open(articles_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            return articles
        else:
            # Create sample custom articles file
            sample_articles = [
                {
                    'title': 'Understanding Volatility Regimes in Modern Markets',
                    'summary': 'A deep dive into how machine learning can help identify different volatility regimes and their impact on portfolio management.',
                    'link': '#',
                    'published': '2024-01-15 10:00',
                    'source': 'VolRegime Research',
                    'category': 'Research',
                    'author': 'VolRegime Team'
                },
                {
                    'title': 'Risk Management in Uncertain Times',
                    'summary': 'Exploring advanced risk metrics and how they can be used to protect portfolios during market turbulence.',
                    'link': '#',
                    'published': '2024-01-14 14:30',
                    'source': 'VolRegime Research',
                    'category': 'Risk Management',
                    'author': 'VolRegime Team'
                }
            ]
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(articles_file), exist_ok=True)
            
            # Save sample articles
            with open(articles_file, 'w', encoding='utf-8') as f:
                json.dump(sample_articles, f, indent=2)
            
            return sample_articles
            
    except Exception as e:
        st.warning(f"Error loading custom articles: {str(e)}")
        return []

def get_all_news(limit_per_source: int = 5) -> List[Dict]:
    """
    Aggregate news from all sources
    
    Args:
        limit_per_source: Maximum articles per news source
    
    Returns:
        Combined list of news articles sorted by date
    """
    all_articles = []
    
    # Get news from different sources
    yahoo_news = get_yahoo_finance_news(limit_per_source)
    marketwatch_news = get_marketwatch_news(limit_per_source)
    custom_articles = get_custom_articles()
    
    # Combine all articles
    all_articles.extend(yahoo_news)
    all_articles.extend(marketwatch_news)
    all_articles.extend(custom_articles)
    
    # Sort by published date (most recent first)
    try:
        all_articles.sort(key=lambda x: datetime.strptime(x['published'], "%Y-%m-%d %H:%M"), reverse=True)
    except:
        # If date parsing fails, keep original order
        pass
    
    return all_articles

def get_stock_specific_news(symbol: str, limit: int = 5) -> List[Dict]:
    """
    Get news articles specific to a stock symbol
    
    Args:
        symbol: Stock ticker symbol
        limit: Maximum number of articles to return
    
    Returns:
        List of stock-specific news articles
    """
    try:
        # Yahoo Finance stock-specific RSS feed
        rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        feed = feedparser.parse(rss_url)
        
        articles = []
        for entry in feed.entries[:limit]:
            try:
                published = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %z")
                published_str = published.strftime("%Y-%m-%d %H:%M")
            except:
                published_str = entry.get('published', 'Unknown date')
            
            article = {
                'title': entry.title,
                'summary': entry.get('summary', entry.get('description', 'No summary available')),
                'link': entry.link,
                'published': published_str,
                'source': 'Yahoo Finance',
                'category': f'{symbol} News',
                'symbol': symbol
            }
            articles.append(article)
        
        return articles
        
    except Exception as e:
        st.warning(f"Error fetching news for {symbol}: {str(e)}")
        return []

def get_fallback_news() -> List[Dict]:
    """
    Provide fallback news articles when RSS feeds are unavailable
    
    Returns:
        List of sample news articles
    """
    return [
        {
            'title': 'Market Volatility Continues Amid Economic Uncertainty',
            'summary': 'Financial markets are experiencing increased volatility as investors navigate uncertain economic conditions and geopolitical tensions.',
            'link': '#',
            'published': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'source': 'Financial News',
            'category': 'Market Update'
        },
        {
            'title': 'Federal Reserve Signals Potential Policy Changes',
            'summary': 'The Federal Reserve is closely monitoring economic indicators and may adjust monetary policy in response to changing market conditions.',
            'link': '#',
            'published': (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"),
            'source': 'Economic News',
            'category': 'Federal Reserve'
        },
        {
            'title': 'Technology Stocks Lead Market Rally',
            'summary': 'Major technology companies are driving market gains as investors show renewed confidence in the sector\'s growth prospects.',
            'link': '#',
            'published': (datetime.now() - timedelta(hours=4)).strftime("%Y-%m-%d %H:%M"),
            'source': 'Tech News',
            'category': 'Technology'
        }
    ]

def clean_html_tags(text: str) -> str:
    """
    Remove HTML tags from text content
    
    Args:
        text: Text that may contain HTML tags
    
    Returns:
        Clean text without HTML tags
    """
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to specified length with ellipsis
    
    Args:
        text: Text to truncate
        max_length: Maximum length of text
    
    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + '...'
