# ui/chat.py
import streamlit as st
from core.ai_services import get_ai_response

def render_chat_tab(df):
    """Renders the main chat interface and handles user interaction."""
    st.subheader(f"Chat with FinBuddy about '{st.session_state.selected_project}'")
    
    # --- Chat History Display ---
    if not st.session_state.messages:
        st.info("Ask me anything about your finances! Try one of the examples below.")
        if st.button("Summarize my spending"):
             st.session_state.prompt_from_button = "Summarize my spending"
             st.rerun()
        if st.button("Create a budget for me"):
            st.session_state.prompt_from_button = "Create a budget for me"
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- User Input ---
    prompt = st.chat_input(f"Ask about '{st.session_state.selected_project}'...")
    
    # Check if a button was pressed
    if "prompt_from_button" in st.session_state and st.session_state.prompt_from_button:
        prompt = st.session_state.prompt_from_button
        st.session_state.prompt_from_button = None # Reset after use

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_ai_response(prompt, df)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()