# ui/sidebar.py
import streamlit as st
import pandas as pd
from datetime import datetime

def render_sidebar():
    """Renders the sidebar for project and data management."""
    with st.sidebar:
        st.header("Project Management")

        # --- Project Creation and Selection ---
        project_options = list(st.session_state.projects.keys())
        try:
            selected_index = project_options.index(st.session_state.selected_project)
        except (ValueError, IndexError):
            selected_index = 0
        
        st.session_state.selected_project = st.selectbox(
            "Current Project", project_options, index=selected_index,
            placeholder="Select or create a project"
        )
        
        # Clear chat history when project changes
        if "last_project" not in st.session_state:
            st.session_state.last_project = st.session_state.selected_project
        if st.session_state.last_project != st.session_state.selected_project:
            st.session_state.messages = []
            st.session_state.last_project = st.session_state.selected_project

        with st.popover("New Project", use_container_width=True):
            with st.form("new_project_form"):
                new_project_name = st.text_input("Project Name").strip()
                if st.form_submit_button("Create"):
                    if new_project_name and new_project_name not in st.session_state.projects:
                        st.session_state.projects[new_project_name] = pd.DataFrame(
                            columns=["date", "category", "amount", "note"]
                        )
                        st.session_state.selected_project = new_project_name
                        # The chat clearing logic above already handles this on the next run
                        st.success(f"Project '{new_project_name}' created!")
                        # NO st.rerun() NEEDED HERE
                    else:
                        st.error("Project name cannot be empty or already exists.")

        st.markdown("---")

        # --- Transaction Management for Selected Project ---
        if st.session_state.selected_project:
            st.subheader(f"Transactions")
            
            with st.popover("➕ Add Transaction", use_container_width=True):
                with st.form("add_txn_form"):
                    date = st.date_input("Date", value=datetime.today())
                    category = st.selectbox("Category", ["Income", "Rent/EMI", "Groceries", "Utilities", "Transport", "Investment", "Other"])
                    amount = st.number_input("Amount (₹)", min_value=0.0, format="%.2f")
                    note = st.text_input("Note (e.g., 'Weekly Groceries')")
                    if st.form_submit_button("Add"):
                        df = st.session_state.projects[st.session_state.selected_project]
                        new_txn = pd.DataFrame([{
                            "date": date.isoformat(),
                            "category": category,
                            "amount": float(amount),
                            "note": note
                        }])
                        st.session_state.projects[st.session_state.selected_project] = pd.concat([df, new_txn], ignore_index=True)
                        st.success("Transaction Added!")
                        # NO st.rerun() NEEDED HERE

            df = st.session_state.projects[st.session_state.selected_project]
            st.dataframe(
                df.sort_values(by="date", ascending=False),
                use_container_width=True,
                hide_index=True
            )