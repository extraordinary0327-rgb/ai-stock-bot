#!/usr/bin/env python3
"""
AI Stock Market Bot
===================
A practical stock analysis & signal bot.

Features:
- Real-time & historical data via Yahoo Finance
- Technical indicators: RSI, SMA, EMA, MACD, Bollinger Bands
- Rule-based AI signals (Buy / Hold / Sell) with confidence
- Multi-stock support
- Clean CLI interface
- Easy to extend with real LLM (OpenAI / Grok / Claude)

Usage:
    python stock_ai_bot.py
    python stock_ai_bot.py AAPL TSLA NVDA NVO AVGO NKE WMT RGTI PLTR DRAM
"""

import sys
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# Try to import ta library for indicators (fallback if not available)
try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False
    print("Warning: 'ta' library not found. Using basic indicators only.")


# ============================================================
# CONFIG
# ============================================================
DEFAULT_TICKERS = ["AVGO", "TSLA", "NVO", "NKE", "WMT", "AAPL", "RGTI", "PLTR", "DRAM"]
LOOKBACK_DAYS = 180          # How much history to fetch
RSI_PERIOD = 14
SMA_SHORT = 20
SMA_LONG = 50
EMA_SHORT = 12
EMA_LONG = 26


# ============================================================
# DATA FETCHING
# ============================================================
def fetch_stock_data(ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
    """Download historical OHLCV data for a ticker."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, auto_adjust=True)
        if df.empty:
            print(f"  [!] No data found for {ticker}")
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"  [!] Error fetching {ticker}: {e}")
        return None


def get_stock_info(ticker: str) -> Dict:
    """Get basic company info and current price."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("shortName") or info.get("longName") or ticker,
            "price": info.get("regularMarketPrice") or info.get("currentPrice"),
            "previous_close": info.get("previousClose"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        return {"name": ticker, "price": None}


# ============================================================
# TECHNICAL INDICATORS
# ============================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataframe."""
    df = df.copy()

    # Simple Moving Averages
    df["SMA_20"] = df["Close"].rolling(window=SMA_SHORT).mean()
    df["SMA_50"] = df["Close"].rolling(window=SMA_LONG).mean()

    # Exponential Moving Averages
    df["EMA_12"] = df["Close"].ewm(span=EMA_SHORT, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=EMA_LONG, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # RSI
    if HAS_TA:
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=RSI_PERIOD).rsi()
    else:
        # Manual RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

    # Volume SMA
    df["Volume_SMA"] = df["Volume"].rolling(window=20).mean()

    return df


# ============================================================
# AI SIGNAL ENGINE (Rule-based + Scoring)
# ============================================================
def generate_signal(df: pd.DataFrame, info: Dict) -> Dict:
    """
    Generate trading signal based on multiple technical factors.
    Returns a dict with signal, confidence, score, and reasons.
    """
    if len(df) < 50:
        return {
            "signal": "HOLD",
            "confidence": 0,
            "score": 0,
            "reasons": ["Insufficient data"],
            "summary": "Not enough historical data for reliable analysis."
        }

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0          # -100 to +100
    reasons = []

    # ----- 1. Trend (SMA crossover) -----
    if latest["SMA_20"] > latest["SMA_50"]:
        score += 20
        reasons.append("Bullish trend: SMA20 above SMA50")
    else:
        score -= 15
        reasons.append("Bearish trend: SMA20 below SMA50")

    # Recent crossover
    if prev["SMA_20"] <= prev["SMA_50"] and latest["SMA_20"] > latest["SMA_50"]:
        score += 15
        reasons.append("Golden Cross detected (SMA20 crossed above SMA50)")
    elif prev["SMA_20"] >= prev["SMA_50"] and latest["SMA_20"] < latest["SMA_50"]:
        score -= 15
        reasons.append("Death Cross detected (SMA20 crossed below SMA50)")

    # ----- 2. RSI -----
    rsi = latest["RSI"]
    if rsi < 30:
        score += 25
        reasons.append(f"RSI oversold ({rsi:.1f}) → potential bounce")
    elif rsi > 70:
        score -= 25
        reasons.append(f"RSI overbought ({rsi:.1f}) → potential pullback")
    elif 40 <= rsi <= 60:
        score += 5
        reasons.append(f"RSI neutral ({rsi:.1f})")
    else:
        reasons.append(f"RSI at {rsi:.1f}")

    # ----- 3. MACD -----
    if latest["MACD"] > latest["MACD_Signal"]:
        score += 15
        reasons.append("MACD above signal line (bullish momentum)")
    else:
        score -= 10
        reasons.append("MACD below signal line (bearish momentum)")

    if prev["MACD"] <= prev["MACD_Signal"] and latest["MACD"] > latest["MACD_Signal"]:
        score += 10
        reasons.append("MACD bullish crossover")
    elif prev["MACD"] >= prev["MACD_Signal"] and latest["MACD"] < latest["MACD_Signal"]:
        score -= 10
        reasons.append("MACD bearish crossover")

    # ----- 4. Bollinger Bands -----
    close = latest["Close"]
    if close < latest["BB_Lower"]:
        score += 15
        reasons.append("Price below lower Bollinger Band (oversold)")
    elif close > latest["BB_Upper"]:
        score -= 15
        reasons.append("Price above upper Bollinger Band (overbought)")
    else:
        reasons.append("Price within Bollinger Bands")

    # ----- 5. Volume -----
    if latest["Volume"] > latest["Volume_SMA"] * 1.5:
        if score > 0:
            score += 8
            reasons.append("High volume confirming upward move")
        else:
            score -= 5
            reasons.append("High volume on downward move")

    # ----- 6. Price vs 52-week range (if available) -----
    if info.get("52w_high") and info.get("52w_low") and close:
        range_pos = (close - info["52w_low"]) / (info["52w_high"] - info["52w_low"])
        if range_pos > 0.9:
            score -= 5
            reasons.append("Near 52-week high")
        elif range_pos < 0.15:
            score += 10
            reasons.append("Near 52-week low (potential value)")

    # Clamp score
    score = max(-100, min(100, score))

    # Determine signal
    if score >= 35:
        signal = "BUY"
        confidence = min(95, 50 + abs(score) // 2)
    elif score <= -30:
        signal = "SELL"
        confidence = min(95, 50 + abs(score) // 2)
    else:
        signal = "HOLD"
        confidence = 40 + abs(score) // 3

    # Summary
    direction = "bullish" if score > 0 else "bearish" if score < 0 else "neutral"
    summary = (
        f"Overall {direction} bias (score: {score:+d}). "
        f"Latest close: ${close:.2f}. "
        f"RSI: {rsi:.1f}."
    )

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
    }


# ============================================================
# ANALYSIS PIPELINE
# ============================================================
def analyze_stock(ticker: str) -> Optional[Dict]:
    """Full analysis pipeline for one stock."""
    print(f"\n{'='*60}")
    print(f"Analyzing {ticker.upper()}...")
    print(f"{'='*60}")

    info = get_stock_info(ticker)
    df = fetch_stock_data(ticker)

    if df is None:
        return None

    df = add_indicators(df)
    result = generate_signal(df, info)

    # Pretty print
    name = info.get("name", ticker)
    price = result.get("price") or info.get("price")
    prev_close = info.get("previous_close")

    change = None
    change_pct = None
    if price and prev_close:
        change = price - prev_close
        change_pct = (change / prev_close) * 100

    print(f"\n  Company     : {name}")
    if price:
        print(f"  Price       : ${price:.2f}", end="")
        if change is not None:
            arrow = "▲" if change >= 0 else "▼"
            print(f"  {arrow} {change:+.2f} ({change_pct:+.2f}%)")
        else:
            print()
    if info.get("sector"):
        print(f"  Sector      : {info['sector']} / {info.get('industry', 'N/A')}")
    if info.get("market_cap"):
        mc = info["market_cap"]
        if mc >= 1e12:
            print(f"  Market Cap  : ${mc/1e12:.2f}T")
        elif mc >= 1e9:
            print(f"  Market Cap  : ${mc/1e9:.1f}B")
        else:
            print(f"  Market Cap  : ${mc/1e6:.0f}M")
    if info.get("pe_ratio"):
        print(f"  P/E Ratio   : {info['pe_ratio']:.1f}")

    print(f"\n  Signal      : {result['signal']}  (Confidence: {result['confidence']}%)")
    print(f"  Score       : {result['score']:+d} / 100")
    print(f"  Summary     : {result['summary']}")
    print(f"\n  Key Reasons:")
    for r in result["reasons"]:
        print(f"    • {r}")

    print(f"\n  Indicators:")
    print(f"    RSI (14)     : {result['rsi']:.1f}")
    print(f"    SMA 20       : ${result['sma20']:.2f}")
    print(f"    SMA 50       : ${result['sma50']:.2f}")
    print(f"    MACD         : {result['macd']:.3f}")

    result["ticker"] = ticker.upper()
    result["name"] = name
    result["info"] = info
    return result


def analyze_multiple(tickers: List[str]) -> List[Dict]:
    """Analyze a list of tickers and return ranked results."""
    results = []
    for t in tickers:
        res = analyze_stock(t.strip().upper())
        if res:
            results.append(res)

    if not results:
        print("\nNo valid results.")
        return []

    # Rank by score
    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n\n{'='*60}")
    print("RANKED SUMMARY")
    print(f"{'='*60}")
    print(f"{'Ticker':<8} {'Signal':<6} {'Score':>6} {'Conf%':>6}  {'Price':>10}  Name")
    print("-" * 70)
    for r in results:
        print(
            f"{r['ticker']:<8} {r['signal']:<6} {r['score']:>+5d} "
            f"{r['confidence']:>5d}%  ${r['price']:>8.2f}  {r['name'][:30]}"
        )

    # Best opportunity
    best = results[0]
    print(f"\n★ Strongest signal: {best['ticker']} → {best['signal']} "
          f"(score {best['score']:+d}, confidence {best['confidence']}%)")

    return results


# ============================================================
# MAIN CLI
# ============================================================
def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║           AI STOCK MARKET BOT  v1.0                      ║
║   Technical Analysis + Multi-Factor Signal Engine        ║
╚══════════════════════════════════════════════════════════╝
    """)


def main():
    print_banner()

    # Get tickers from command line or use defaults
    if len(sys.argv) > 1:
        tickers = [t.upper() for t in sys.argv[1:]]
    else:
        print("No tickers provided. Using default watchlist:")
        print(" ", ", ".join(DEFAULT_TICKERS))
        print("\nTip: python stock_ai_bot.py AAPL TSLA NVDA")
        tickers = DEFAULT_TICKERS

    print(f"\nFetching data for {len(tickers)} stock(s)...\n")

    results = analyze_multiple(tickers)

    print(f"\n\nAnalysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Disclaimer: This is for educational purposes only. Not financial advice.")
    print("Always do your own research and consider risk management.\n")


if __name__ == "__main__":
    main()
