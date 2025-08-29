# core/analytics.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px

@st.cache_data(show_spinner="Forecasting future expenses...")
def get_forecast(df: pd.DataFrame) -> str:
    """
    Generates a financial forecast summary using a simple linear trend model.
    This provides a more robust forecast than the previous version.
    """
    if df.empty or len(df) < 3:
        return "Not enough transaction data to generate a forecast. Please add more entries."
    
    # Filter out income to forecast expenses only
    expense_df = df[df['category'] != 'Income'].copy()
    if len(expense_df) < 3:
        return "Not enough expense data to generate a forecast. Please add more expense entries."

    expense_df["date"] = pd.to_datetime(expense_df["date"])
    
    # Aggregate expenses by day
    daily_expenses = expense_df.groupby(expense_df["date"].dt.date)["amount"].sum().reset_index()
    daily_expenses["ordinal"] = pd.to_datetime(daily_expenses["date"]).map(datetime.toordinal)

    # Simple linear regression
    coeffs = np.polyfit(daily_expenses["ordinal"], daily_expenses["amount"], 1)
    poly = np.poly1d(coeffs)
    
    last_day_ord = daily_expenses["ordinal"].max()
    future_ord = np.arange(last_day_ord + 1, last_day_ord + 31)
    
    # Predict future daily expenses and ensure they are not negative
    future_preds = poly(future_ord)
    future_preds[future_preds < 0] = 0
    
    total_forecast = float(np.sum(future_preds))
    avg_daily_spend = expense_df.groupby(expense_df['date'].dt.date)['amount'].sum().mean()
    
    return (
        f"### 🗓️ 30-Day Expense Forecast\n\n"
        f"Based on your recent spending habits, you are projected to spend approximately **₹{total_forecast:,.2f}** over the next 30 days.\n\n"
        f"*Your average daily spend has been **₹{avg_daily_spend:,.2f}**.*\n\n"
        f"> **Disclaimer:** This is a simple trend-based projection and may not account for large, irregular expenses."
    )

@st.cache_data
def create_spending_pie_chart(df: pd.DataFrame):
    """Creates a pie chart of spending by category."""
    expense_df = df[df['category'] != 'Income']
    if expense_df.empty:
        return None
    fig = px.pie(expense_df, names='category', values='amount', title='Spending by Category',
                 hole=0.3, color_discrete_sequence=px.colors.sequential.RdBu)
    fig.update_layout(legend_title_text='Categories')
    return fig

@st.cache_data
def create_income_expense_bar_chart(df: pd.DataFrame):
    """Creates a bar chart comparing total income vs. total expenses."""
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M').astype(str)
    summary = df.groupby('month').apply(lambda x: pd.Series({
        'Income': x[x['category'] == 'Income']['amount'].sum(),
        'Expense': x[x['category'] != 'Income']['amount'].sum()
    })).reset_index()
    
    if summary.empty:
        return None

    summary_melted = summary.melt(id_vars='month', var_name='type', value_name='amount', value_vars=['Income', 'Expense'])
    fig = px.bar(summary_melted, x='month', y='amount', color='type',
                 title='Monthly Income vs. Expenses', barmode='group',
                 labels={'amount': 'Amount (₹)', 'month': 'Month', 'type': 'Transaction Type'},
                 color_discrete_map={'Income': '#4CAF50', 'Expense': '#F44336'})
    return fig