"""
Sentiment Analysis Module for AI Stock Market Bot
-------------------------------------------------
Fetches recent news / headlines for a ticker and scores sentiment.

Sources (in order of preference):
1. Yahoo Finance news (via yfinance) - no key needed
2. Google News RSS - no key needed
3. Optional: NewsAPI / Finnhub if API keys are set

Scoring:
- VADER (good for social/financial short text)
- TextBlob (fallback)
- Combined compound score from -1 (very bearish) to +1 (very bullish)
"""

import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import feedparser
import yfinance as yf

# Sentiment libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER = SentimentIntensityAnalyzer()
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False


# ============================================================
# NEWS FETCHERS
# ============================================================

def fetch_yahoo_news(ticker: str, max_items: int = 12) -> List[Dict]:
    """Fetch recent news from Yahoo Finance via yfinance."""
    items = []
    try:
        stock = yf.Ticker(ticker)
        news = stock.news or []
        for n in news[:max_items]:
            # yfinance news structure can vary
            content = n.get("content") or n
            title = (
                content.get("title")
                or n.get("title")
                or content.get("headline")
                or ""
            )
            summary = (
                content.get("summary")
                or content.get("description")
                or n.get("summary")
                or ""
            )
            publisher = (
                content.get("provider", {}).get("displayName")
                if isinstance(content.get("provider"), dict)
                else content.get("publisher") or n.get("publisher") or "Yahoo"
            )
            link = (
                content.get("canonicalUrl", {}).get("url")
                if isinstance(content.get("canonicalUrl"), dict)
                else content.get("link") or n.get("link") or ""
            )
            pub_time = content.get("pubDate") or n.get("providerPublishTime") or ""

            if title:
                items.append({
                    "title": title.strip(),
                    "summary": (summary or "")[:300].strip(),
                    "publisher": publisher,
                    "link": link,
                    "published": str(pub_time),
                    "source": "yahoo",
                })
    except Exception as e:
        print(f"  [yahoo news] {e}")
    return items


def fetch_google_news_rss(ticker: str, company_name: str = "", max_items: int = 10) -> List[Dict]:
    """Fetch news via Google News RSS (no API key)."""
    items = []
    queries = [ticker]
    if company_name and company_name.lower() != ticker.lower():
        # Clean company name for query
        clean = re.sub(r"[^\w\s]", "", company_name).strip()
        if clean:
            queries.append(f'"{clean}" stock')

    for q in queries:
        try:
            url = f"https://news.google.com/rss/search?q={q.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_items]:
                title = entry.get("title", "").strip()
                summary = entry.get("summary", "") or entry.get("description", "")
                # Clean HTML from summary
                summary = re.sub(r"<[^>]+>", "", summary)[:300]
                if title and not any(i["title"] == title for i in items):
                    items.append({
                        "title": title,
                        "summary": summary.strip(),
                        "publisher": entry.get("source", {}).get("title", "Google News"),
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "source": "google_rss",
                    })
        except Exception as e:
            print(f"  [google rss] {e}")
        if len(items) >= max_items:
            break
    return items[:max_items]


def fetch_finnhub_news(ticker: str, max_items: int = 10) -> List[Dict]:
    """Optional: Finnhub news (free tier, needs FINNHUB_API_KEY)."""
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return []

    items = []
    try:
        import requests
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        url = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={ticker}&from={from_date}&to={to_date}&token={api_key}"
        )
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            for n in resp.json()[:max_items]:
                items.append({
                    "title": n.get("headline", ""),
                    "summary": (n.get("summary") or "")[:300],
                    "publisher": n.get("source", "Finnhub"),
                    "link": n.get("url", ""),
                    "published": str(n.get("datetime", "")),
                    "source": "finnhub",
                })
    except Exception as e:
        print(f"  [finnhub] {e}")
    return items


def fetch_newsapi(ticker: str, company_name: str = "", max_items: int = 10) -> List[Dict]:
    """Optional: NewsAPI.org (free tier, needs NEWSAPI_KEY)."""
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return []

    items = []
    try:
        import requests
        q = ticker if not company_name else f"{ticker} OR {company_name}"
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={q}&language=en&sortBy=publishedAt&pageSize={max_items}&apiKey={api_key}"
        )
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            for n in resp.json().get("articles", [])[:max_items]:
                items.append({
                    "title": n.get("title") or "",
                    "summary": (n.get("description") or "")[:300],
                    "publisher": n.get("source", {}).get("name", "NewsAPI"),
                    "link": n.get("url", ""),
                    "published": n.get("publishedAt", ""),
                    "source": "newsapi",
                })
    except Exception as e:
        print(f"  [newsapi] {e}")
    return items


def collect_news(ticker: str, company_name: str = "", max_items: int = 15) -> List[Dict]:
    """Aggregate news from multiple free + optional paid sources."""
    all_items = []
    seen_titles = set()

    sources = [
        lambda: fetch_yahoo_news(ticker, max_items=10),
        lambda: fetch_google_news_rss(ticker, company_name, max_items=10),
        lambda: fetch_finnhub_news(ticker, max_items=8),
        lambda: fetch_newsapi(ticker, company_name, max_items=8),
    ]

    for src in sources:
        try:
            for item in src():
                title_key = item["title"].lower()[:80]
                if title_key and title_key not in seen_titles:
                    seen_titles.add(title_key)
                    all_items.append(item)
        except Exception:
            continue

    return all_items[:max_items]


# ============================================================
# SENTIMENT SCORING
# ============================================================

def score_text(text: str) -> Dict[str, float]:
    """
    Score a single piece of text.
    Returns dict with compound, pos, neu, neg (VADER-style).
    """
    if not text or not text.strip():
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}

    text = text.strip()

    if HAS_VADER:
        scores = VADER.polarity_scores(text)
        return {
            "compound": scores["compound"],
            "pos": scores["pos"],
            "neu": scores["neu"],
            "neg": scores["neg"],
        }

    if HAS_TEXTBLOB:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        # Approximate VADER-style
        if polarity > 0.1:
            return {"compound": polarity, "pos": polarity, "neu": 1 - abs(polarity), "neg": 0.0}
        elif polarity < -0.1:
            return {"compound": polarity, "pos": 0.0, "neu": 1 - abs(polarity), "neg": abs(polarity)}
        else:
            return {"compound": polarity, "pos": 0.0, "neu": 1.0, "neg": 0.0}

    return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}


def analyze_news_sentiment(news_items: List[Dict]) -> Dict:
    """
    Score a list of news items and return aggregate sentiment.
    """
    if not news_items:
        return {
            "compound": 0.0,
            "label": "NEUTRAL",
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "total": 0,
            "avg_compound": 0.0,
            "top_positive": [],
            "top_negative": [],
            "items": [],
        }

    scored = []
    for item in news_items:
        text = f"{item.get('title', '')}. {item.get('summary', '')}"
        s = score_text(text)
        entry = {**item, **s}
        scored.append(entry)

    compounds = [x["compound"] for x in scored]
    avg = sum(compounds) / len(compounds)

    positive = [x for x in scored if x["compound"] >= 0.15]
    negative = [x for x in scored if x["compound"] <= -0.15]
    neutral = [x for x in scored if -0.15 < x["compound"] < 0.15]

    # Label
    if avg >= 0.25:
        label = "BULLISH"
    elif avg >= 0.08:
        label = "SLIGHTLY BULLISH"
    elif avg <= -0.25:
        label = "BEARISH"
    elif avg <= -0.08:
        label = "SLIGHTLY BEARISH"
    else:
        label = "NEUTRAL"

    # Top headlines
    top_pos = sorted(positive, key=lambda x: x["compound"], reverse=True)[:3]
    top_neg = sorted(negative, key=lambda x: x["compound"])[:3]

    return {
        "compound": round(avg, 4),
        "label": label,
        "positive": len(positive),
        "negative": len(negative),
        "neutral": len(neutral),
        "total": len(scored),
        "avg_compound": round(avg, 4),
        "top_positive": top_pos,
        "top_negative": top_neg,
        "items": scored,
    }


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def get_sentiment(ticker: str, company_name: str = "") -> Dict:
    """
    Full sentiment pipeline for a ticker.
    Returns aggregated sentiment + scored headlines.
    """
    ticker = ticker.upper().strip()

    # Try to get company name if not provided
    if not company_name:
        try:
            info = yf.Ticker(ticker).info
            company_name = info.get("shortName") or info.get("longName") or ""
        except Exception:
            company_name = ""

    news = collect_news(ticker, company_name)
    result = analyze_news_sentiment(news)
    result["ticker"] = ticker
    result["company_name"] = company_name
    result["news_count"] = len(news)
    return result


def sentiment_to_score_boost(sentiment: Dict) -> Tuple[int, str]:
    """
    Convert sentiment result into a score adjustment for the main signal engine.
    Returns (boost points, reason string)
    """
    compound = sentiment.get("compound", 0.0)
    total = sentiment.get("total", 0)

    if total < 2:
        return 0, "Insufficient news for sentiment"

    if compound >= 0.35:
        return 12, f"Strongly bullish news sentiment ({compound:+.2f})"
    elif compound >= 0.15:
        return 7, f"Bullish news sentiment ({compound:+.2f})"
    elif compound >= 0.05:
        return 3, f"Slightly positive news ({compound:+.2f})"
    elif compound <= -0.35:
        return -12, f"Strongly bearish news sentiment ({compound:+.2f})"
    elif compound <= -0.15:
        return -7, f"Bearish news sentiment ({compound:+.2f})"
    elif compound <= -0.05:
        return -3, f"Slightly negative news ({compound:+.2f})"
    else:
        return 0, f"Neutral news sentiment ({compound:+.2f})"


def print_sentiment_report(result: Dict):
    """Pretty print sentiment analysis."""
    print(f"\n{'='*60}")
    print(f"SENTIMENT ANALYSIS: {result.get('ticker', '?')}")
    if result.get("company_name"):
        print(f"  {result['company_name']}")
    print(f"{'='*60}")
    print(f"Overall          : {result['label']}  (compound: {result['compound']:+.3f})")
    print(f"Headlines scored : {result['total']}")
    print(f"  Positive       : {result['positive']}")
    print(f"  Negative       : {result['negative']}")
    print(f"  Neutral        : {result['neutral']}")

    if result.get("top_positive"):
        print(f"\nTop Positive Headlines:")
        for h in result["top_positive"]:
            print(f"  + [{h['compound']:+.2f}] {h['title'][:90]}")

    if result.get("top_negative"):
        print(f"\nTop Negative Headlines:")
        for h in result["top_negative"]:
            print(f"  - [{h['compound']:+.2f}] {h['title'][:90]}")

    boost, reason = sentiment_to_score_boost(result)
    print(f"\nSignal adjustment: {boost:+d} points ({reason})")


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    print(f"Fetching news & analyzing sentiment for {ticker}...")
    result = get_sentiment(ticker)
    print_sentiment_report(result)
