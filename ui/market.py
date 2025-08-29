# ui/market.py
import streamlit as st
from core.stock_market import get_stock_data, create_price_chart, get_stock_analysis

def render_market_tab():
    """Renders the UI for the stock market information tab."""
    st.subheader("📈 Stock Market Information")
    
    ticker = st.text_input("Enter Stock Ticker Symbol", 
                           placeholder="e.g., AAPL, GOOG, or RELIANCE.NS for Indian stocks").upper()

    if st.button("Get Stock Data", use_container_width=True):
        if ticker:
            with st.spinner(f"Fetching data for {ticker}..."):
                info, history = get_stock_data(ticker)
                st.session_state.stock_info = info
                st.session_state.stock_history = history
        else:
            st.warning("Please enter a stock ticker symbol.")
            st.session_state.stock_info = None
            st.session_state.stock_history = None

    if 'stock_info' in st.session_state and st.session_state.stock_info:
        info = st.session_state.stock_info
        history = st.session_state.stock_history

        st.markdown(f"## {info.get('longName', 'N/A')} ({info.get('symbol', 'N/A')})")
        
        # --- Key Metrics ---
        price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        prev_close = info.get('previousClose', 0)
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close != 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Last Price", f"{price:,.2f} {info.get('currency', '')}", f"{change:,.2f} ({change_pct:.2f}%)")
        col2.metric("Market Cap", f"{(info.get('marketCap', 0) / 1e9):.2f}B")
        col3.metric("Volume", f"{info.get('volume', 0):,}")

        st.markdown("---")
        
        # --- Chart and Analysis ---
        chart = create_price_chart(history, ticker)
        st.plotly_chart(chart, use_container_width=True)

        if st.button("Generate AI Analysis", use_container_width=True):
            with st.spinner("🤖 FinBuddy is analyzing the data..."):
                analysis = get_stock_analysis(info)
                st.session_state.stock_analysis = analysis
        
        if 'stock_analysis' in st.session_state and st.session_state.stock_analysis:
            st.markdown("### AI-Powered Fundamental Snapshot")
            st.info(st.session_state.stock_analysis)
    
    elif 'stock_info' in st.session_state and st.session_state.stock_info is None:
        st.error(f"Could not find data. Please check the symbol (e.g., 'MSFT' for US stocks, 'RELIANCE.NS' for Indian stocks).")