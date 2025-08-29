# core/ai_services.py
import streamlit as st
import os
import pandas as pd
import google.generativeai as genai
from gradio_client import Client
from google.api_core import exceptions
from .analytics import get_forecast  # Import analytics functions

# --- API & MODEL SETUP ---
try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        st.error("Missing Google API Key. Please set it in your secrets or environment variables.")
        st.stop()
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    # This is where Granite is used for one specific, lightweight task.
    granite_client = Client("ad1xya/granite-api")
except Exception as e:
    st.error(f"Error during AI model initialization: {e}")
    st.stop()


@st.cache_data(show_spinner=False)
def get_transaction_category(description: str) -> str:
    """Uses the specialized Granite model for simple transaction categorization."""
    prompt = f"Categorize the following transaction into one of these: Income, Rent/EMI, Groceries, Utilities, Transport, Investment, Other. Transaction: '{description}'"
    try:
        # This is the single, mandated use of the Granite model.
        return granite_client.predict(prompt=prompt, api_name="/predict")
    except Exception:
        return "Other" # Fallback on error


@st.cache_data(show_spinner=False)
def get_ai_response(user_query: str, df: pd.DataFrame) -> str:
    """
    Intelligent router to classify user intent and generate a response using the appropriate tool or model.
    """
    # Prepare context from the DataFrame
    context_snippet = df.sort_values(by="date", ascending=False).head(20).to_string(index=False) if not df.empty else "No transactions yet."

    # 1. Intent Classification using Gemini
    classifier_prompt = f"""
    You are an intelligent routing system for a financial assistant app. Classify the user's request into ONE of the following categories based on their query. Respond with only the category name.

    Categories:
    - 'BUDGETING': For questions about creating or analyzing a spending budget.
    - 'FORECASTING': For questions about future spending projections or predictions.
    - 'CATEGORIZATION': For direct requests to categorize a single transaction description (e.g., "categorize starbucks coffee").
    - 'SAVINGS_INVESTMENT': For questions about saving money, investment plans, or financial goals.
    - 'TAX_INFO': For questions related to income tax, deductions, or tax planning.
    - 'CASH_MANAGEMENT': For questions about cash flow, income vs. expenses, or managing money.
    - 'REPORT_SUMMARY': For requests to summarize spending, find top expenses, or general data analysis.
    - 'GENERAL_QUERY': For all other financial questions, advice, or conversations.

    User request: "{user_query}"
    Category:
    """
    try:
        response = gemini_model.generate_content(classifier_prompt, generation_config={"max_output_tokens": 20})
        intent = response.text.strip().upper()

        # 2. Tool Routing based on intent
        if "CATEGORIZATION" in intent:
            description = user_query.lower().replace("categorize", "").strip()
            category = get_transaction_category(description)
            return f"The transaction '{description}' is best categorized as: **{category}**"

        elif "FORECASTING" in intent:
            return get_forecast(df)

        # For all other intents, use Gemini with a specialized prompt
        else:
            system_prompts = {
                "BUDGETING": "You are a helpful financial advisor. Create a simple, actionable monthly budget based on the user's request and their transaction history. Present it clearly in markdown.",
                "SAVINGS_INVESTMENT": "You are a financial planning assistant. Provide guidance on savings strategies or investment options based on the user's query and financial data. Include a disclaimer that this is not professional financial advice.",
                "TAX_INFO": "You are a tax information assistant for India. Answer user questions about income tax concepts clearly and concisely. Include a disclaimer that you are not a certified tax professional and the user should consult one for official advice.",
                "CASH_MANAGEMENT": "You are a financial analyst. Explain concepts of cash flow and analyze the provided transaction data to give insights on managing income and expenses.",
                "REPORT_SUMMARY": "You are a data analyst. Summarize the provided transaction history, highlighting key insights like top spending categories, spending trends, and total income vs. expenses. Use markdown for clear formatting.",
                "GENERAL_QUERY": "You are FinBuddy, a helpful and friendly financial assistant. Answer the user's question concisely and clearly based on their request and the provided context of their recent financial transactions."
            }
            system_prompt = system_prompts.get(intent, system_prompts["GENERAL_QUERY"])
            
            full_prompt = f"{system_prompt}\n\nUser Request: {user_query}\n\nTransaction History (for context):\n{context_snippet}"
            
            ai_response = gemini_model.generate_content(full_prompt)
            return ai_response.text

    except exceptions.GoogleAPICallError as e:
        return f"Sorry, there was an issue with the AI service: API call failed. Details: {e}"
    except Exception as e:
        return f"An unexpected error occurred while processing your request: {e}"