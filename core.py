"""
Core shared functions for AI Stock Market Bot
"""

import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False


def fetch_stock_data(ticker: str, period: str = "1y") -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Download historical OHLCV + info."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, auto_adjust=True)
        if df.empty:
            return None, None
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        info = stock.info
        return df, info
    except Exception:
        return None, None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators."""
    df = df.copy()

    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    if HAS_TA:
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        df["ATR"] = ta.volatility.AverageTrueRange(
            df["High"], df["Low"], df["Close"], window=14
        ).average_true_range()
    else:
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df["ATR"] = true_range.rolling(14).mean()

    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
    df["Volume_SMA"] = df["Volume"].rolling(window=20).mean()

    return df


def generate_signal(df: pd.DataFrame, info: Optional[dict] = None) -> Dict:
    """Multi-factor signal engine. Returns signal dict."""
    info = info or {}
    if len(df) < 50:
        return {
            "signal": "HOLD", "confidence": 0, "score": 0,
            "reasons": ["Insufficient data"], "summary": "Not enough data.",
            "price": float(df["Close"].iloc[-1]) if len(df) else 0,
            "rsi": 50.0, "sma20": 0.0, "sma50": 0.0,
            "macd": 0.0, "macd_signal": 0.0, "atr": 0.0,
        }

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    reasons = []

    # Trend
    if latest["SMA_20"] > latest["SMA_50"]:
        score += 18
        reasons.append("✅ Bullish trend: SMA20 > SMA50")
    else:
        score -= 14
        reasons.append("❌ Bearish trend: SMA20 < SMA50")

    if prev["SMA_20"] <= prev["SMA_50"] and latest["SMA_20"] > latest["SMA_50"]:
        score += 15
        reasons.append("🌟 Golden Cross detected!")
    elif prev["SMA_20"] >= prev["SMA_50"] and latest["SMA_20"] < latest["SMA_50"]:
        score -= 15
        reasons.append("💀 Death Cross detected!")

    if not pd.isna(latest.get("SMA_200")):
        if latest["Close"] > latest["SMA_200"]:
            score += 10
            reasons.append("✅ Price above 200-day SMA")
        else:
            score -= 10
            reasons.append("❌ Price below 200-day SMA")

    # RSI
    rsi = latest["RSI"]
    if rsi < 30:
        score += 25
        reasons.append(f"🔥 RSI oversold ({rsi:.1f})")
    elif rsi > 70:
        score -= 25
        reasons.append(f"⚠️ RSI overbought ({rsi:.1f})")
    elif 40 <= rsi <= 60:
        score += 5
        reasons.append(f"➖ RSI neutral ({rsi:.1f})")
    else:
        reasons.append(f"RSI at {rsi:.1f}")

    # MACD
    if latest["MACD"] > latest["MACD_Signal"]:
        score += 14
        reasons.append("✅ MACD above signal")
    else:
        score -= 10
        reasons.append("❌ MACD below signal")

    if prev["MACD"] <= prev["MACD_Signal"] and latest["MACD"] > latest["MACD_Signal"]:
        score += 12
        reasons.append("🌟 MACD bullish crossover")
    elif prev["MACD"] >= prev["MACD_Signal"] and latest["MACD"] < latest["MACD_Signal"]:
        score -= 12
        reasons.append("💀 MACD bearish crossover")

    # Bollinger
    close = latest["Close"]
    if close < latest["BB_Lower"]:
        score += 15
        reasons.append("🔥 Below lower Bollinger Band")
    elif close > latest["BB_Upper"]:
        score -= 15
        reasons.append("⚠️ Above upper Bollinger Band")
    else:
        reasons.append("Price inside Bollinger Bands")

    # Volume
    if latest["Volume"] > latest["Volume_SMA"] * 1.5:
        if score > 0:
            score += 8
            reasons.append("📈 High volume confirmation")
        else:
            score -= 6
            reasons.append("📉 High volume on down move")

    # 52-week
    high_52 = info.get("fiftyTwoWeekHigh")
    low_52 = info.get("fiftyTwoWeekLow")
    if high_52 and low_52 and close:
        range_pos = (close - low_52) / (high_52 - low_52)
        if range_pos > 0.92:
            score -= 6
            reasons.append("Near 52-week high")
        elif range_pos < 0.12:
            score += 12
            reasons.append("Near 52-week low")

    score = max(-100, min(100, int(score)))

    if score >= 35:
        signal = "BUY"
        confidence = min(95, 55 + abs(score) // 2)
    elif score <= -30:
        signal = "SELL"
        confidence = min(95, 55 + abs(score) // 2)
    else:
        signal = "HOLD"
        confidence = 40 + abs(score) // 3

    direction = "bullish" if score > 10 else "bearish" if score < -10 else "neutral"
    summary = f"Overall {direction} bias (score: {score:+d}). Close: ${close:.2f} | RSI: {rsi:.1f}"

    return {
        "signal": signal,
        "confidence": confidence,
        "score": score,
        "reasons": reasons,
        "summary": summary,
        "price": float(close),
        "rsi": float(rsi),
        "sma20": float(latest["SMA_20"]),
        "sma50": float(latest["SMA_50"]),
        "macd": float(latest["MACD"]),
        "macd_signal": float(latest["MACD_Signal"]),
        "atr": float(latest.get("ATR", 0) or 0),
    }


# Popular universes for scanner
SP500_SAMPLE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
    "V", "XOM", "JPM", "WMT", "MA", "PG", "HD", "CVX", "MRK", "ABBV",
    "KO", "PEP", "AVGO", "COST", "MCD", "CSCO", "TMO", "ACN", "DHR", "ABT",
    "WFC", "LIN", "ADBE", "CRM", "NKE", "TXN", "PM", "NEE", "ORCL", "AMD",
    "INTC", "QCOM", "IBM", "AMAT", "NOW", "INTU", "ISRG", "BKNG", "SBUX", "GE",
]

TECH_SECTOR = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "AVGO",
    "ORCL", "CRM", "ADBE", "CSCO", "IBM", "QCOM", "TXN", "AMAT", "NOW", "INTU",
    "PANW", "SNOW", "PLTR", "UBER", "SHOP", "SQ", "COIN", "NET", "DDOG", "CRWD",
]

FINANCE_SECTOR = [
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
    "PNC", "TFC", "COF", "BK", "STT", "AIG", "MET", "PRU", "ALL", "TRV",
]

HEALTHCARE_SECTOR = [
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "PFE", "DHR", "BMY",
    "AMGN", "GILD", "MDT", "ISRG", "SYK", "BSX", "VRTX", "REGN", "CI", "ELV",
]


# ============================================================
# FULL ANALYSIS WITH SENTIMENT
# ============================================================

def analyze_with_sentiment(ticker: str, period: str = "1y", include_sentiment: bool = True) -> Dict:
    """
    Full analysis: technicals + optional news sentiment.
    """
    from sentiment import get_sentiment, sentiment_to_score_boost

    df, info = fetch_stock_data(ticker, period)
    if df is None:
        return {"error": f"Could not fetch data for {ticker}"}

    df = add_indicators(df)
    tech = generate_signal(df, info)
    tech["ticker"] = ticker.upper()
    tech["name"] = (info or {}).get("shortName") or ticker

    sentiment = None
    if include_sentiment:
        try:
            company = (info or {}).get("shortName") or (info or {}).get("longName") or ""
            sentiment = get_sentiment(ticker, company)
            boost, reason = sentiment_to_score_boost(sentiment)
            # Adjust score
            original_score = tech["score"]
            tech["score"] = max(-100, min(100, original_score + boost))
            tech["sentiment_boost"] = boost
            tech["sentiment_reason"] = reason

            # Re-derive signal from new score
            score = tech["score"]
            if score >= 35:
                tech["signal"] = "BUY"
                tech["confidence"] = min(95, 55 + abs(score) // 2)
            elif score <= -30:
                tech["signal"] = "SELL"
                tech["confidence"] = min(95, 55 + abs(score) // 2)
            else:
                tech["signal"] = "HOLD"
                tech["confidence"] = 40 + abs(score) // 3

            tech["reasons"].append(f"📰 {reason} ({boost:+d})")
            tech["summary"] = (
                f"Overall {'bullish' if score > 10 else 'bearish' if score < -10 else 'neutral'} "
                f"bias (score: {score:+d}). Close: ${tech['price']:.2f} | RSI: {tech['rsi']:.1f} | "
                f"News: {sentiment.get('label', 'N/A')}"
            )
        except Exception as e:
            tech["sentiment_error"] = str(e)

    return {
        "tech": tech,
        "sentiment": sentiment,
        "info": info,
        "df": df,
    }
