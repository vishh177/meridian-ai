import yfinance as yf
import pandas as pd
import numpy as np

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)

def calculate_macd(prices):
    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    histogram = macd - signal
    return {
        "macd": round(macd.iloc[-1], 4),
        "signal": round(signal.iloc[-1], 4),
        "histogram": round(histogram.iloc[-1], 4),
        "crossover": "bullish" if macd.iloc[-1] > signal.iloc[-1] else "bearish"
    }

def calculate_bollinger_bands(prices, period=20):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    current_price = prices.iloc[-1]
    position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
    if position > 0.8:
        signal = "overbought"
    elif position < 0.2:
        signal = "oversold"
    else:
        signal = "neutral"
    return {
        "upper": round(upper.iloc[-1], 2),
        "middle": round(sma.iloc[-1], 2),
        "lower": round(lower.iloc[-1], 2),
        "signal": signal,
        "position": round(position * 100, 1)
    }

def calculate_trend(prices):
    sma50 = prices.rolling(window=50).mean().iloc[-1]
    sma200 = prices.rolling(window=200).mean().iloc[-1]
    current = prices.iloc[-1]
    if current > sma50 > sma200:
        trend = "strong uptrend"
    elif current > sma50:
        trend = "uptrend"
    elif current < sma50 < sma200:
        trend = "strong downtrend"
    elif current < sma50:
        trend = "downtrend"
    else:
        trend = "sideways"
    return {
        "trend": trend,
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "above_sma50": bool(current > sma50),
        "above_sma200": bool(current > sma200)
    }

def get_technical_analysis(ticker):
    stock = yf.Ticker(ticker)
    history = stock.history(period="1y")
    prices = history["Close"]

    rsi = calculate_rsi(prices)
    macd = calculate_macd(prices)
    bollinger = calculate_bollinger_bands(prices)
    trend = calculate_trend(prices)

    if rsi > 70:
        rsi_signal = "overbought"
    elif rsi < 30:
        rsi_signal = "oversold"
    else:
        rsi_signal = "neutral"

    bullish_signals = sum([
        macd["crossover"] == "bullish",
        bollinger["signal"] == "oversold",
        trend["above_sma50"],
        trend["above_sma200"],
        rsi_signal == "oversold"
    ])

    bearish_signals = sum([
        macd["crossover"] == "bearish",
        bollinger["signal"] == "overbought",
        not trend["above_sma50"],
        not trend["above_sma200"],
        rsi_signal == "overbought"
    ])

    score = round((bullish_signals / (bullish_signals + bearish_signals)) * 100) if (bullish_signals + bearish_signals) > 0 else 50

    return {
        "rsi": rsi,
        "rsi_signal": rsi_signal,
        "macd": macd,
        "bollinger": bollinger,
        "trend": trend,
        "technical_score": score,
        "bullish_signals": bullish_signals,
        "bearish_signals": bearish_signals
    }