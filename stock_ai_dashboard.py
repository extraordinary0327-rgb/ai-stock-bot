#!/usr/bin/env python3
"""
AI Stock Market Bot - Web Dashboard
===================================
Beautiful Streamlit interface for the AI Stock Market Bot.

Run with:
    streamlit run stock_ai_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Stock Market Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .signal-buy {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
    }
    .signal-sell {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
    }
    .signal-hold {
        background: linear-gradient(135deg, #4b6cb7, #182848);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #333;
    }
    .stMetric {
        background: #1a1a2e;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA & INDICATORS
# ============================================================
@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, period: str = "1y"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, auto_adjust=True)
        if df.empty:
            return None, None
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)

        info = stock.info
        return df, info
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return None, None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Moving Averages
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # RSI
    if HAS_TA:
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    else:
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

    # Volume
    df["Volume_SMA"] = df["Volume"].rolling(window=20).mean()

    # ATR for volatility
    if HAS_TA:
        df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    else:
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df["ATR"] = true_range.rolling(14).mean()

    return df


def generate_signal(df: pd.DataFrame, info: dict) -> dict:
    if len(df) < 50:
        return {
            "signal": "HOLD", "confidence": 0, "score": 0,
            "reasons": ["Insufficient data"], "summary": "Not enough data."
        }

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    reasons = []

    # 1. Trend - SMA
    if latest["SMA_20"] > latest["SMA_50"]:
        score += 18
        reasons.append("✅ Bullish trend: SMA20 > SMA50")
    else:
        score -= 14
        reasons.append("❌ Bearish trend: SMA20 < SMA50")

    # Golden / Death Cross
    if prev["SMA_20"] <= prev["SMA_50"] and latest["SMA_20"] > latest["SMA_50"]:
        score += 15
        reasons.append("🌟 Golden Cross detected!")
    elif prev["SMA_20"] >= prev["SMA_50"] and latest["SMA_20"] < latest["SMA_50"]:
        score -= 15
        reasons.append("💀 Death Cross detected!")

    # 200-day MA (long-term trend)
    if not pd.isna(latest.get("SMA_200")):
        if latest["Close"] > latest["SMA_200"]:
            score += 10
            reasons.append("✅ Price above 200-day SMA (long-term bullish)")
        else:
            score -= 10
            reasons.append("❌ Price below 200-day SMA (long-term bearish)")

    # 2. RSI
    rsi = latest["RSI"]
    if rsi < 30:
        score += 25
        reasons.append(f"🔥 RSI oversold ({rsi:.1f}) → bounce potential")
    elif rsi > 70:
        score -= 25
        reasons.append(f"⚠️ RSI overbought ({rsi:.1f}) → pullback risk")
    elif 40 <= rsi <= 60:
        score += 5
        reasons.append(f"➖ RSI neutral ({rsi:.1f})")
    else:
        reasons.append(f"RSI at {rsi:.1f}")

    # 3. MACD
    if latest["MACD"] > latest["MACD_Signal"]:
        score += 14
        reasons.append("✅ MACD above signal (bullish momentum)")
    else:
        score -= 10
        reasons.append("❌ MACD below signal (bearish momentum)")

    if prev["MACD"] <= prev["MACD_Signal"] and latest["MACD"] > latest["MACD_Signal"]:
        score += 12
        reasons.append("🌟 MACD bullish crossover")
    elif prev["MACD"] >= prev["MACD_Signal"] and latest["MACD"] < latest["MACD_Signal"]:
        score -= 12
        reasons.append("💀 MACD bearish crossover")

    # 4. Bollinger Bands
    close = latest["Close"]
    if close < latest["BB_Lower"]:
        score += 15
        reasons.append("🔥 Price below lower Bollinger Band (oversold)")
    elif close > latest["BB_Upper"]:
        score -= 15
        reasons.append("⚠️ Price above upper Bollinger Band (overbought)")
    else:
        reasons.append("Price inside Bollinger Bands")

    # 5. Volume
    if latest["Volume"] > latest["Volume_SMA"] * 1.5:
        if score > 0:
            score += 8
            reasons.append("📈 High volume confirming move")
        else:
            score -= 6
            reasons.append("📉 High volume on down move")

    # 6. 52-week range
    high_52 = info.get("fiftyTwoWeekHigh")
    low_52 = info.get("fiftyTwoWeekLow")
    if high_52 and low_52 and close:
        range_pos = (close - low_52) / (high_52 - low_52)
        if range_pos > 0.92:
            score -= 6
            reasons.append("Near 52-week high")
        elif range_pos < 0.12:
            score += 12
            reasons.append("Near 52-week low (potential value zone)")

    score = max(-100, min(100, score))

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
        "atr": float(latest.get("ATR", 0)),
    }


# ============================================================
# CHARTS
# ============================================================
def create_candlestick_chart(df: pd.DataFrame, ticker: str):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{ticker} Price", "RSI", "MACD")
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350"
    ), row=1, col=1)

    # SMAs
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20",
                             line=dict(color="#2196F3", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50",
                             line=dict(color="#FF9800", width=1.5)), row=1, col=1)
    if "SMA_200" in df.columns and not df["SMA_200"].isna().all():
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_200"], name="SMA 200",
                                 line=dict(color="#9C27B0", width=1.5, dash="dot")), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
                             line=dict(color="rgba(100,100,100,0.3)", width=1), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
                             line=dict(color="rgba(100,100,100,0.3)", width=1),
                             fill="tonexty", fillcolor="rgba(100,100,100,0.1)", showlegend=False), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                             line=dict(color="#7E57C2", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # MACD
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="MACD Hist",
                         marker_color=colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                             line=dict(color="#2196F3", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                             line=dict(color="#FF9800", width=1.5)), row=3, col=1)

    fig.update_layout(
        height=750,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    return fig


# ============================================================
# MAIN APP
# ============================================================
def main():
    st.markdown('<p class="main-header">📈 AI Stock Market Bot</p>', unsafe_allow_html=True)
    st.caption("Technical Analysis + Multi-Factor Signal Engine | Educational use only")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        ticker = st.text_input("Stock Ticker", value="NVDA").upper().strip()
        period = st.selectbox("History Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)

        st.markdown("---")
        st.subheader("Quick Watchlist")
        watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD", "INTC", "SPY"]
        selected = st.multiselect("Select stocks to rank", watchlist, default=["NVDA", "AAPL", "TSLA", "MSFT"])

        st.markdown("---")
        st.info("💡 This bot uses pure technical analysis. Always do your own research.")
        st.caption("Not financial advice.")

    # Main analysis
    if not ticker:
        st.warning("Please enter a ticker symbol.")
        return

    with st.spinner(f"Analyzing {ticker}..."):
        df, info = fetch_stock_data(ticker, period)

    if df is None or info is None:
        st.error(f"Could not fetch data for **{ticker}**. Check the ticker symbol.")
        return

    df = add_indicators(df)
    result = generate_signal(df, info)

    # Header metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    name = info.get("shortName") or info.get("longName") or ticker
    price = result["price"]
    prev_close = info.get("previousClose")
    change = price - prev_close if prev_close else 0
    change_pct = (change / prev_close * 100) if prev_close else 0

    with col1:
        st.metric("Price", f"${price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    with col2:
        signal_class = f"signal-{result['signal'].lower()}"
        st.markdown(f'<div class="{signal_class}">{result["signal"]}</div>', unsafe_allow_html=True)
        st.caption(f"Confidence: {result['confidence']}%")
    with col3:
        st.metric("Score", f"{result['score']:+d}", "out of ±100")
    with col4:
        st.metric("RSI (14)", f"{result['rsi']:.1f}")
    with col5:
        mc = info.get("marketCap")
        if mc:
            if mc >= 1e12:
                mc_str = f"${mc/1e12:.2f}T"
            elif mc >= 1e9:
                mc_str = f"${mc/1e9:.1f}B"
            else:
                mc_str = f"${mc/1e6:.0f}M"
            st.metric("Market Cap", mc_str)
        else:
            st.metric("Market Cap", "N/A")

    # Company info
    st.markdown(f"**{name}** · {info.get('sector', 'N/A')} · {info.get('industry', 'N/A')}")

    # Chart
    st.plotly_chart(create_candlestick_chart(df, ticker), use_container_width=True)

    # Analysis details
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🧠 Signal Reasoning")
        for reason in result["reasons"]:
            st.markdown(f"- {reason}")
        st.info(result["summary"])

    with col_right:
        st.subheader("📊 Key Indicators")
        ind_data = {
            "Indicator": ["RSI (14)", "SMA 20", "SMA 50", "MACD", "MACD Signal", "ATR (14)"],
            "Value": [
                f"{result['rsi']:.1f}",
                f"${result['sma20']:.2f}",
                f"${result['sma50']:.2f}",
                f"{result['macd']:.3f}",
                f"{result['macd_signal']:.3f}",
                f"{result['atr']:.2f}",
            ]
        }
        st.dataframe(pd.DataFrame(ind_data), use_container_width=True, hide_index=True)

        # Additional fundamentals
        st.subheader("🏢 Fundamentals")
        fund_cols = st.columns(2)
        with fund_cols[0]:
            pe = info.get("trailingPE")
            st.write(f"**P/E Ratio:** {pe:.1f}" if pe else "**P/E Ratio:** N/A")
            st.write(f"**52W High:** ${info.get('fiftyTwoWeekHigh', 0):.2f}")
        with fund_cols[1]:
            st.write(f"**Forward P/E:** {info.get('forwardPE', 'N/A')}")
            st.write(f"**52W Low:** ${info.get('fiftyTwoWeekLow', 0):.2f}")

    # Watchlist ranking
    if selected:
        st.markdown("---")
        st.subheader("📋 Watchlist Ranking")

        ranking = []
        progress = st.progress(0)
        for i, t in enumerate(selected):
            d, inf = fetch_stock_data(t, "6mo")
            if d is not None:
                d = add_indicators(d)
                r = generate_signal(d, inf or {})
                ranking.append({
                    "Ticker": t,
                    "Signal": r["signal"],
                    "Score": r["score"],
                    "Confidence": f"{r['confidence']}%",
                    "Price": f"${r['price']:.2f}",
                    "RSI": f"{r['rsi']:.1f}",
                    "Name": (inf or {}).get("shortName", t)
                })
            progress.progress((i + 1) / len(selected))
        progress.empty()

        if ranking:
            rank_df = pd.DataFrame(ranking).sort_values("Score", ascending=False)
            st.dataframe(rank_df, use_container_width=True, hide_index=True)

            best = rank_df.iloc[0]
            st.success(f"★ Strongest signal: **{best['Ticker']}** → **{best['Signal']}** (Score: {best['Score']:+d})")

    # Footer
    st.markdown("---")
    st.caption(f"Data as of {datetime.now().strftime('%Y-%m-%d %H:%M')} | Built with yfinance + technical analysis | Not financial advice")


if __name__ == "__main__":
    main()
