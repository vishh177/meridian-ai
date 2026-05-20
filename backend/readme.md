# MERIDIAN — AI Investment Research Terminal

Multi-agent AI system that analyses stocks like a hedge fund analyst.

## Features
- Real-time stock data (Yahoo Finance)
- Fundamental Agent — financial analysis
- Sentiment Agent — news analysis  
- Technical Agent — RSI, MACD, Bollinger Bands
- FinBERT NLP — transformer-based sentiment model
- LSTM Neural Network — 7-day price forecasting
- SEC EDGAR — real filing analysis
- Stock comparison mode

## Tech Stack
Backend: Python, FastAPI, PyTorch, HuggingFace Transformers
Frontend: React
AI: Groq (LLaMA 3.3), FinBERT, LSTM

## Setup
1. Clone the repo
2. Add `.env` file with GROQ_API_KEY and NEWSAPI_KEY
3. pip install -r requirements.txt
4. uvicorn main:app --reload
5. cd frontend && npm install && npm start