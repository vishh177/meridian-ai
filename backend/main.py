from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from groq import Groq
import yfinance as yf
import requests
import json
import os
from technical import get_technical_analysis
from lstm import run_lstm_prediction
from finbert import get_finbert_score
from sec_edgar import get_sec_analysis

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "name": info.get("longName", ticker),
        "price": info.get("currentPrice"),
        "pe_ratio": info.get("trailingPE"),
        "revenue_growth": info.get("revenueGrowth"),
        "gross_margins": info.get("grossMargins"),
        "debt_to_equity": info.get("debtToEquity"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "market_cap": info.get("marketCap"),
        "analyst_target": info.get("targetMeanPrice"),
        "recommendation": info.get("recommendationKey"),
    }

def get_news(ticker):
    api_key = os.getenv("NEWSAPI_KEY")
    url = f"https://newsapi.org/v2/everything?q={ticker}+stock&language=en&sortBy=publishedAt&pageSize=20&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    
    filtered = []
    for a in articles:
        title = a.get("title") or ""
        description = a.get("description") or ""
        text = title + " " + description
        if text.strip() and all(ord(c) < 128 for c in text[:50]):
            filtered.append({
                "title": title,
                "description": description
            })
        if len(filtered) == 10:
            break
    
    return filtered

def run_fundamental_agent(stock_data):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a fundamental analyst at a hedge fund. Analyse the financial data and return a JSON with: score (0-100), strengths (list of 3), weaknesses (list of 3). Return JSON only."},
            {"role": "user", "content": f"Analyse this stock data: {json.dumps(stock_data)}"}
        ]
    )
    return response.choices[0].message.content

def run_sentiment_agent(news):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a sentiment analyst at a hedge fund. Analyse these news articles and return a JSON with: score (0-100), sentiment (bullish/bearish/neutral), key_themes (list of 3), red_flags (list of 2). Return JSON only."},
            {"role": "user", "content": f"Analyse this news: {json.dumps(news)}"}
        ]
    )
    return response.choices[0].message.content

def run_portfolio_manager(ticker, stock_data, fundamental, sentiment):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a portfolio manager at a hedge fund. Based on fundamental and sentiment analysis, produce a full investment memo. Return JSON with: verdict (Strong Buy/Buy/Hold/Sell/Strong Sell), confidence (0-100), bull_case (list of 3), bear_case (list of 3), price_target, catalysts (list of 3), summary (2 sentences)."},
            {"role": "user", "content": f"Stock: {ticker}\nData: {json.dumps(stock_data)}\nFundamental: {fundamental}\nSentiment: {sentiment}"}
        ]
    )
    return response.choices[0].message.content

@app.post("/analyse")
@app.post("/analyse")
def analyse_stock(data: dict):
    try:
        ticker = data["ticker"].upper()
        
        stock_data = get_stock_data(ticker)
        news = get_news(ticker)
        technical = get_technical_analysis(ticker)
        fundamental = run_fundamental_agent(stock_data)
        sentiment = run_sentiment_agent(news)
        memo = run_portfolio_manager(ticker, stock_data, fundamental, sentiment)
        lstm = run_lstm_prediction(ticker)
        finbert_sentiment = get_finbert_score(news)
        sec = get_sec_analysis(ticker)

        return {
    "ticker": ticker,
    "stock_data": stock_data,
    "technical": technical,
    "fundamental": fundamental,
    "sentiment": sentiment,
    "finbert": finbert_sentiment,
    "memo": memo,
    "lstm": lstm,
    "sec": sec
}
    
    except Exception as e:
        return {"error": str(e)}
@app.post("/compare")
def compare_stocks(data: dict):
    try:
        ticker1 = data["ticker1"].upper()
        ticker2 = data["ticker2"].upper()

        stock1 = get_stock_data(ticker1)
        stock2 = get_stock_data(ticker2)

        tech1 = get_technical_analysis(ticker1)
        tech2 = get_technical_analysis(ticker2)

        fund1 = run_fundamental_agent(stock1)
        fund2 = run_fundamental_agent(stock2)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are a portfolio manager comparing two stocks. 
                    Return JSON only with:
                    {
                        "winner": "ticker of better investment",
                        "winner_reason": "2 sentence explanation",
                        "ticker1_score": 0-100,
                        "ticker2_score": 0-100,
                        "verdict": "one line comparison summary",
                        "better_value": "ticker with better valuation",
                        "better_growth": "ticker with better growth",
                        "better_momentum": "ticker with better technical momentum",
                        "recommendation": "which to buy and why in 2 sentences"
                    }"""
                },
                {
                    "role": "user",
                    "content": f"Compare {ticker1}: {json.dumps(stock1)} with technical {json.dumps(tech1)} vs {ticker2}: {json.dumps(stock2)} with technical {json.dumps(tech2)}"
                }
            ]
        )

        comparison = response.choices[0].message.content
        clean = comparison.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)

        return {
            "ticker1": ticker1,
            "ticker2": ticker2,
            "stock1": stock1,
            "stock2": stock2,
            "technical1": tech1,
            "technical2": tech2,
            "fundamental1": fund1,
            "fundamental2": fund2,
            "comparison": parsed
        }
    except Exception as e:
        return {"error": str(e)}
@app.post("/chat")
def chat(data: dict):
    try:
        message = data["message"]
        context = data.get("context", {})
        history = data.get("history", [])

        system_prompt = f"""You are MERIDIAN, an AI investment research assistant. 
You have just completed a full analysis of {context.get('ticker', 'a stock')}.

Here is the analysis context:
- Current Price: ${context.get('price')}
- Verdict: {context.get('verdict')}
- Confidence: {context.get('confidence')}%
- Fundamental Score: {context.get('fundamental_score')}/100
- Sentiment Score: {context.get('sentiment_score')}/100
- Technical Score: {context.get('technical_score')}/100
- RSI: {context.get('rsi')} ({context.get('rsi_signal')})
- Trend: {context.get('trend')}
- LSTM 7-day prediction: {context.get('lstm_change')}%
- Bull case: {context.get('bull_case')}
- Bear case: {context.get('bear_case')}
- SEC Filing Score: {context.get('sec_score')}/100
- Management Tone: {context.get('management_tone')}

Answer questions about this stock analysis clearly and concisely.
Be direct. Use specific numbers from the analysis.
Never make up data not in the context.
Keep responses under 150 words."""

        messages = [{"role": "system", "content": system_prompt}]
        for h in history:
            messages.append(h)
        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )

        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}