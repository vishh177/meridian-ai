import requests
import json
from groq import Groq
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

HEADERS = {"User-Agent": "stock-analyst-app contact@example.com"}

def get_cik(ticker):
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS)
    data = response.json()
    for key in data:
        if data[key]["ticker"].upper() == ticker.upper():
            cik = str(data[key]["cik_str"]).zfill(10)
            return cik
    return None

def get_latest_filing(cik, form_type="10-Q"):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    data = response.json()
    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    dates = filings.get("filingDate", [])
    accessions = filings.get("accessionNumber", [])
    primary_docs = filings.get("primaryDocument", [])
    for i, form in enumerate(forms):
        if form == form_type:
            return {
                "accession": accessions[i],
                "date": dates[i],
                "primary_doc": primary_docs[i],
                "cik": cik
            }
    return None

def get_filing_text(filing):
    cik = int(filing["cik"])
    accession_clean = filing["accession"].replace("-", "")
    doc = filing["primary_doc"]
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{doc}"
    response = requests.get(doc_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return text[:8000]

def analyse_filing(ticker, text, filing_date):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a senior equity analyst at a hedge fund reading an SEC filing. 
                Extract and analyse the most important financial information. 
                Return JSON only with these fields:
                {
                    "revenue_trend": "string describing revenue trend",
                    "margin_trend": "string describing margin trend",
                    "key_risks": ["risk1", "risk2", "risk3"],
                    "key_opportunities": ["opp1", "opp2", "opp3"],
                    "management_tone": "bullish/neutral/bearish",
                    "red_flags": ["flag1", "flag2"],
                    "filing_score": 0-100,
                    "summary": "2 sentence summary"
                }"""
            },
            {
                "role": "user",
                "content": f"Analyse this SEC filing for {ticker} filed on {filing_date}:\n\n{text}"
            }
        ]
    )
    return response.choices[0].message.content

def get_sec_analysis(ticker):
    try:
        print(f"Fetching SEC filing for {ticker}...")
        cik = get_cik(ticker)
        if not cik:
            return {"error": "CIK not found for ticker"}
        filing = get_latest_filing(cik, "10-Q")
        if not filing:
            filing = get_latest_filing(cik, "10-K")
        if not filing:
            return {"error": "No filing found"}
        text = get_filing_text(filing)
        if not text:
            return {"error": "Could not extract filing text"}
        analysis = analyse_filing(ticker, text, filing["date"])
        clean = analysis.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        return {
            "filing_type": "10-Q",
            "filing_date": filing["date"],
            **parsed
        }
    except Exception as e:
        return {"error": str(e)}