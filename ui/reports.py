# ui/reports.py
import streamlit as st
import pandas as pd
from core.analytics import create_spending_pie_chart, create_income_expense_bar_chart

def render_reports_tab(df: pd.DataFrame):
    """Renders the dashboard with visual reports."""
    st.subheader("Financial Dashboard")

    if df.empty:
        st.warning("No transaction data available. Please add transactions in the sidebar to view reports.")
        return

    # --- Key Metrics ---
    total_income = df[df['category'] == 'Income']['amount'].sum()
    total_expense = df[df['category'] != 'Income']['amount'].sum()
    net_savings = total_income - total_expense

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Income", f"₹{total_income:,.2f}", delta_color="normal")
    col2.metric("Total Expenses", f"₹{total_expense:,.2f}", delta_color="inverse")
    col3.metric("Net Savings", f"₹{net_savings:,.2f}", delta=f"{net_savings:,.2f}")

    st.markdown("---")

    # --- Visualizations ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Spending by Category")
        pie_chart = create_spending_pie_chart(df)
        if pie_chart:
            st.plotly_chart(pie_chart, use_container_width=True)
        else:
            st.info("No expense data to display.")

    with col2:
        st.subheader("Income vs. Expenses")
        bar_chart = create_income_expense_bar_chart(df)
        if bar_chart:
            st.plotly_chart(bar_chart, use_container_width=True)
        else:
            st.info("No income/expense data to display.")