import yfinance as yf
import numpy as np
import pandas as pd

def run_backtest(ticker, period="6mo"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    spy = yf.Ticker("SPY")
    spy_hist = spy.history(period=period)

    prices = hist["Close"]
    volumes = hist["Volume"]

    # ── CALCULATE SIGNALS ──
    # RSI
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9).mean()

    # Bollinger Bands
    sma20 = prices.rolling(20).mean()
    std20 = prices.rolling(20).std()
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20

    # SMA trend
    sma50 = prices.rolling(50).mean()

    # ── GENERATE SIGNALS ──
    signals = pd.Series(index=prices.index, dtype=float)
    for i in range(1, len(prices)):
        bullish = 0
        bearish = 0

        if rsi.iloc[i] < 35:
            bullish += 2
        elif rsi.iloc[i] > 65:
            bearish += 2

        if macd.iloc[i] > signal_line.iloc[i] and macd.iloc[i-1] <= signal_line.iloc[i-1]:
            bullish += 2
        elif macd.iloc[i] < signal_line.iloc[i] and macd.iloc[i-1] >= signal_line.iloc[i-1]:
            bearish += 2

        if prices.iloc[i] <= lower_band.iloc[i]:
            bullish += 1
        elif prices.iloc[i] >= upper_band.iloc[i]:
            bearish += 1

        if prices.iloc[i] > sma50.iloc[i]:
            bullish += 1
        else:
            bearish += 1

        if bullish >= 3:
            signals.iloc[i] = 1
        elif bearish >= 3:
            signals.iloc[i] = -1
        else:
            signals.iloc[i] = 0

    # ── SIMULATE TRADES ──
    cash = 10000
    shares = 0
    position = 0
    trades = []
    portfolio_values = []
    buy_price = 0

    for i in range(len(prices)):
        price = prices.iloc[i]
        sig = signals.iloc[i] if i < len(signals) else 0
        date = str(prices.index[i].date())

        if sig == 1 and position == 0:
            shares = cash / price
            cash = 0
            position = 1
            buy_price = price
            trades.append({
                "date": date,
                "action": "BUY",
                "price": round(float(price), 2),
                "shares": round(float(shares), 4)
            })

        elif sig == -1 and position == 1:
            cash = shares * price
            profit_pct = ((price - buy_price) / buy_price) * 100
            trades.append({
                "date": date,
                "action": "SELL",
                "price": round(float(price), 2),
                "profit_pct": round(float(profit_pct), 2),
                "profit_loss": "WIN" if profit_pct > 0 else "LOSS"
            })
            shares = 0
            position = 0

        portfolio_value = cash + (shares * price)
        portfolio_values.append(portfolio_value)

    # Close any open position
    if position == 1:
        final_price = prices.iloc[-1]
        cash = shares * final_price
        profit_pct = ((final_price - buy_price) / buy_price) * 100
        trades.append({
            "date": str(prices.index[-1].date()),
            "action": "SELL (CLOSE)",
            "price": round(float(final_price), 2),
            "profit_pct": round(float(profit_pct), 2),
            "profit_loss": "WIN" if profit_pct > 0 else "LOSS"
        })

    # ── CALCULATE METRICS ──
    portfolio_series = pd.Series(portfolio_values)
    total_return = ((cash - 10000) / 10000) * 100

    buy_hold_return = ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]) * 100
    spy_return = ((spy_hist["Close"].iloc[-1] - spy_hist["Close"].iloc[0]) / spy_hist["Close"].iloc[0]) * 100

    daily_returns = portfolio_series.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

    rolling_max = portfolio_series.cummax()
    drawdown = (portfolio_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    sell_trades = [t for t in trades if t["action"] in ["SELL", "SELL (CLOSE)"]]
    wins = len([t for t in sell_trades if t.get("profit_loss") == "WIN"])
    win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0

    return {
        "ticker": ticker,
        "period": period,
        "initial_capital": 10000,
        "final_value": round(float(cash), 2),
        "total_return": round(float(total_return), 2),
        "buy_hold_return": round(float(buy_hold_return), 2),
        "spy_return": round(float(spy_return), 2),
        "beat_market": bool(total_return > spy_return),
        "sharpe_ratio": round(float(sharpe), 3),
        "max_drawdown": round(float(max_drawdown), 2),
        "total_trades": len(sell_trades),
        "win_rate": round(float(win_rate), 2),
        "wins": wins,
        "losses": len(sell_trades) - wins,
        "trades": trades[-10:]
    }