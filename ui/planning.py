# ui/planning.py
import streamlit as st
import pandas as pd
from core.analytics import get_forecast
from core.ai_services import get_ai_response

def render_planning_tab(df: pd.DataFrame):
    """Renders the planning tab for forecasts and budgets."""
    st.subheader("Financial Planning Tools")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Expense Forecaster")
        with st.container(border=True):
            forecast_result = get_forecast(df)
            st.markdown(forecast_result)

    with col2:
        st.subheader("AI Budget Assistant")
        with st.container(border=True):
            st.markdown("Generate a personalized budget based on your spending.")
            if st.button("Generate My Budget Plan"):
                if 'budget_plan' not in st.session_state or st.session_state.budget_plan is None:
                    with st.spinner("Creating your budget..."):
                        prompt = "Based on my transaction history, create a simple monthly budget plan for me. Suggest categories and amounts."
                        st.session_state.budget_plan = get_ai_response(prompt, df)
                
            if 'budget_plan' in st.session_state and st.session_state.budget_plan:
                st.markdown(st.session_state.budget_plan)