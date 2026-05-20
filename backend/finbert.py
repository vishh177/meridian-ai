from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

MODEL_NAME = "ProsusAI/finbert"
tokenizer = None
model = None

def load_finbert():
    global tokenizer, model
    if tokenizer is None:
        print("Loading FinBERT model...")
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()
        print("FinBERT loaded.")

def analyse_sentiment(texts):
    load_finbert()
    results = []
    for text in texts:
        if not text or len(text.strip()) < 10:
            continue
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).numpy()[0]
        labels = ["positive", "negative", "neutral"]
        sentiment = labels[np.argmax(probs)]
        confidence = float(np.max(probs))
        results.append({
            "text": text[:100],
            "sentiment": sentiment,
            "confidence": round(confidence, 3),
            "positive": round(float(probs[0]), 3),
            "negative": round(float(probs[1]), 3),
            "neutral": round(float(probs[2]), 3),
        })
    return results

def get_finbert_score(news_articles):
    texts = [
        (a.get("title") or "") + " " + (a.get("description") or "")
        for a in news_articles
    ]
    texts = [t.strip() for t in texts if len(t.strip()) > 10]
    if not texts:
        return {"score": 50, "sentiment": "neutral", "breakdown": []}
    results = analyse_sentiment(texts)
    if not results:
        return {"score": 50, "sentiment": "neutral", "breakdown": []}
    positive = sum(1 for r in results if r["sentiment"] == "positive")
    negative = sum(1 for r in results if r["sentiment"] == "negative")
    neutral = sum(1 for r in results if r["sentiment"] == "neutral")
    total = len(results)
    weighted_score = sum(
        r["positive"] * 100 if r["sentiment"] == "positive"
        else r["negative"] * 0 if r["sentiment"] == "negative"
        else 50
        for r in results
    ) / total
    if weighted_score >= 60:
        overall = "bullish"
    elif weighted_score <= 40:
        overall = "bearish"
    else:
        overall = "neutral"
    return {
        "score": round(weighted_score),
        "sentiment": overall,
        "positive_count": positive,
        "negative_count": negative,
        "neutral_count": neutral,
        "total_articles": total,
        "breakdown": results[:5]
    }