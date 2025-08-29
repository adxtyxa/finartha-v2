# core/stock_market.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from .ai_services import gemini_model

@st.cache_data(ttl=300) # Cache data for 5 minutes
def get_stock_data(ticker_symbol: str):
    """Fetches real-time data for a given stock ticker."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        # Check if the market price is available, a good indicator of a valid ticker
        if 'currentPrice' not in info and 'regularMarketPrice' not in info:
            return None, f"Could not find data for ticker '{ticker_symbol}'. Please check the symbol (e.g., 'MSFT' for US, 'RELIANCE.NS' for India)."
        hist = ticker.history(period="1y")
        return info, hist
    except Exception as e:
        return None, f"An error occurred while fetching data: {e}"

def create_price_chart(history_df: pd.DataFrame, ticker_symbol: str):
    """Creates a Plotly chart for the stock's historical price."""
    fig = px.line(history_df, x=history_df.index, y="Close", 
                  title=f"{ticker_symbol} - 1 Year Stock Price",
                  labels={'Close': 'Closing Price (in currency)', 'Date': 'Date'})
    fig.update_layout(template="plotly_white")
    return fig

def get_stock_analysis(info: dict):
    """
    Uses Gemini to generate a neutral, data-driven analysis of a stock.
    This prompt is engineered to be safe and avoid giving direct advice.
    """
    # Sanitize the info dict to avoid sending too much data
    relevant_info = {
        "Company Name": info.get("longName"),
        "Symbol": info.get("symbol"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "Business Summary": info.get("longBusinessSummary"),
        "Current Price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "Market Cap": info.get("marketCap"),
        "52-Week High": info.get("fiftyTwoWeekHigh"),
        "52-Week Low": info.get("fiftyTwoWeekLow"),
        "Volume": info.get("volume"),
        "P/E Ratio": info.get("trailingPE"),
    }

    prompt = f"""
    You are a financial data analyst. Your task is to provide a neutral, objective summary of the following company based on the data provided.
    
    **Instructions:**
    1.  Start with a brief, one-paragraph summary of the company's business.
    2.  Present the key financial metrics in a structured way (e.g., bullet points).
    3.  Provide a "Fundamental Snapshot" section that briefly analyzes what the metrics might indicate (e.g., "A high P/E ratio can suggest that investors expect higher earnings growth in the future").
    4.  **DO NOT** use words like "buy," "sell," "hold," "good investment," "bad investment," or any other advisory language.
    5.  **DO NOT** predict the stock's future price.
    6.  Conclude with a clear, mandatory disclaimer: "Disclaimer: This is an AI-generated analysis based on public data and is for informational purposes only. It is not financial advice. Please consult with a qualified financial advisor before making any investment decisions."

    **Stock Data:**
    {relevant_info}
    """

    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred during AI analysis: {e}"