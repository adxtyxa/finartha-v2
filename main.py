# app.py
import streamlit as st
import os
import pandas as pd
import numpy as np
from datetime import datetime
import google.generativeai as genai
from gradio_client import Client
from google.api_core import exceptions

# ---------------------------
# 1. CONFIGURATION & INITIALIZATION
# ---------------------------

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="FinBuddy AI Assistant",
    layout="centered",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# --- API & MODEL SETUP ---
try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        st.error("Missing Google API Key. Please set it in your secrets or environment variables.")
        st.stop()
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    granite_client = Client("ad1xya/granite-api")
except Exception as e:
    st.error(f"Error during model initialization: {e}")
    st.stop()

# --- STYLING ---
st.markdown("""
<style>
    /* Clean up sidebar */
    [data-testid="stSidebar"] {
        background-color: #071018;
        padding: 10px;
    }
    /* Chat bubbles */
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Main app background */
    .stApp {
        background-color: #0b0f15;
        color: #e6eef6;
    }
    .stButton>button {
        background-color:#11151a; color:#e6eef6; border:1px solid #233044;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------
# 2. SESSION STATE MANAGEMENT
# ---------------------------

def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    defaults = {
        "projects": {},
        "selected_project": None,
        "messages": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ---------------------------
# 3. BACKEND CORE LOGIC (TOOLS)
# ---------------------------

@st.cache_data
def get_transaction_category(description: str) -> str:
    """Uses a specialized model (Granite) for simple transaction categorization."""
    prompt = f"Categorize the following transaction into one of these: Income, Rent/EMI, Groceries, Utilities, Transport, Investment, Other. Transaction: '{description}'"
    try:
        return granite_client.predict(prompt=prompt, api_name="/predict")
    except Exception:
        return "Other" # Fallback on error

@st.cache_data
def get_forecast(df: pd.DataFrame) -> str:
    """Generates a financial forecast summary."""
    if df.empty or len(df) < 2:
        return "Not enough transaction data to generate a forecast. Please add more entries."
    
    daily = df.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily_totals = daily.groupby(daily["date"].dt.date)["amount"].sum().reset_index()
    daily_totals["ordinal"] = pd.to_datetime(daily_totals["date"]).map(datetime.toordinal)

    coeffs = np.polyfit(daily_totals["ordinal"], daily_totals["amount"], 1)
    poly = np.poly1d(coeffs)
    future_ord = np.arange(daily_totals["ordinal"].max() + 1, daily_totals["ordinal"].max() + 31)
    preds = poly(future_ord)
    preds[preds < 0] = 0
    total_forecast = float(np.sum(preds))
    
    return f"Based on a simple trend analysis of your recent transactions, you are projected to spend approximately **₹{total_forecast:,.2f}** over the next 30 days."

def get_ai_response(user_query: str, project_context: str) -> str:
    """
    Acts as an intelligent router. It first classifies the user's intent
    and then generates a response using the appropriate tool.
    """
    # 1. Intent Classification
    classifier_prompt = f"""
    You are a financial assistant's routing system. Classify the user's request into one of the following categories: 'BUDGET', 'FORECAST', 'CATEGORIZE', 'GENERAL'.
    
    - 'BUDGET': For questions about creating a spending budget.
    - 'FORECAST': For questions about future spending projections.
    - 'CATEGORIZE': For direct requests to categorize a single transaction description (e.g., "categorize starbucks coffee").
    - 'GENERAL': For all other financial questions, summaries, or advice.
    
    User request: "{user_query}"
    Category:
    """
    try:
        # Use a lightweight call for classification
        response = gemini_model.generate_content(classifier_prompt, generation_config={"max_output_tokens": 10})
        intent = response.text.strip().upper()

        # 2. Tool Routing
        if "BUDGET" in intent:
            prompt = f"You are a helpful financial advisor. Based on the user's request and their recent transaction history, create a simple, actionable monthly budget. Present it clearly in markdown. \n\nUser Request: {user_query}\n\nTransaction History:\n{project_context}"
            return gemini_model.generate_content(prompt).text

        elif "FORECAST" in intent:
            df = st.session_state.projects.get(st.session_state.selected_project)
            if df is not None:
                return get_forecast(df)
            return "Please select a project with transactions to generate a forecast."

        elif "CATEGORIZE" in intent:
            # Extract the description from the query
            description = user_query.lower().replace("categorize", "").strip()
            return f"The transaction '{description}' is best categorized as: **{get_transaction_category(description)}**"

        else: # GENERAL
            prompt = f"You are FinBuddy, a helpful and friendly financial assistant. Answer the user's question based on their request and the provided context of their recent financial transactions. Be concise and clear. \n\nUser Request: {user_query}\n\nTransaction History:\n{project_context}"
            return gemini_model.generate_content(prompt).text
            
    except exceptions.GoogleAPICallError as e:
        return f"Sorry, there was an issue with the AI service: API call failed. Details: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# ---------------------------
# 4. UI: SIDEBAR
# ---------------------------
def render_sidebar():
    """Renders the sidebar for project and data management."""
    with st.sidebar:
        st.header("Project Management")

        # --- Project Creation and Selection ---
        project_options = list(st.session_state.projects.keys())
        selected_index = project_options.index(st.session_state.selected_project) if st.session_state.selected_project in project_options else 0
        
        st.session_state.selected_project = st.selectbox(
            "Current Project", project_options, index=selected_index, placeholder="Select or create a project"
        )

        with st.popover("New Project", use_container_width=True):
            with st.form("new_project_form"):
                new_project_name = st.text_input("Project Name", key="new_project_name")
                if st.form_submit_button("Create"):
                    if new_project_name and new_project_name not in st.session_state.projects:
                        st.session_state.projects[new_project_name] = pd.DataFrame(columns=["date", "category", "amount", "note"])
                        st.session_state.selected_project = new_project_name
                        st.success(f"Project '{new_project_name}' created!")
                        st.rerun()
                    else:
                        st.error("Name cannot be empty or already exists.")

        # --- Transaction Management for Selected Project ---
        if st.session_state.selected_project:
            st.markdown("---")
            st.subheader(f"Transactions for '{st.session_state.selected_project}'")
            
            with st.popover("Add Transaction", use_container_width=True):
                with st.form("add_txn_form"):
                    date = st.date_input("Date", value=datetime.today())
                    category = st.selectbox("Category", ["Income", "Rent/EMI", "Groceries", "Utilities", "Transport", "Investment", "Other"])
                    amount = st.number_input("Amount (₹)", min_value=0.0, format="%.2f")
                    note = st.text_input("Note (optional)")
                    if st.form_submit_button("Add"):
                        df = st.session_state.projects[st.session_state.selected_project]
                        new_txn = pd.DataFrame([{"date": date.isoformat(), "category": category, "amount": float(amount), "note": note}])
                        st.session_state.projects[st.session_state.selected_project] = pd.concat([df, new_txn], ignore_index=True)
                        st.success("Transaction Added!")
                        st.rerun()

            # Display Transactions
            df = st.session_state.projects[st.session_state.selected_project]
            st.dataframe(df.sort_values(by="date", ascending=False), use_container_width=True)


# ---------------------------
# 5. UI: MAIN CHAT INTERFACE
# ---------------------------
def render_chat_interface():
    """Renders the main chat interface and handles user interaction."""
    st.title("FinBuddy AI Assistant")

    # --- Landing Page / Welcome Message ---
    if not st.session_state.selected_project:
        st.info("👋 Welcome to FinBuddy! Create or select a project in the sidebar to get started.")
        st.markdown("""
        **Here are some things you can ask:**
        - "What were my top 3 spending categories last month?"
        - "Create a budget for me based on my recent spending."
        - "How much am I likely to spend in the next 30 days?"
        - "Categorize a transaction for 'movie tickets'."
        """)
        return

    # --- Chat History Display ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- User Input ---
    if prompt := st.chat_input(f"Ask about '{st.session_state.selected_project}'..."):
        # Add user message to state and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Prepare context for the AI
                df = st.session_state.projects[st.session_state.selected_project]
                context_df = df.sort_values(by="date", ascending=False).head(20)
                context_snippet = context_df.to_string(index=False) if not context_df.empty else "No transactions yet."
                
                response = get_ai_response(prompt, context_snippet)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

# ---------------------------
# 6. APP ENTRYPOINT
# ---------------------------
if __name__ == "__main__":
    initialize_session_state()
    render_sidebar()
    render_chat_interface()