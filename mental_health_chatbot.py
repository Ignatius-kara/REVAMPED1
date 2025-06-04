import streamlit as st
import json
import re
from datetime import datetime
import pandas as pd

# Load the dataset
@st.cache_data
def load_dataset():
    dataset = []
    with open("optimized_mental_health_chatbot_dataset.jsonl", "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

# Match user input to an intent
def match_intent(user_message, dataset):
    user_message = user_message.lower().strip()
    for entry in dataset:
        # Check if user message matches any pattern for the intent
        patterns = [entry["user_message"].lower()]  # Simplified pattern matching
        for pattern in patterns:
            if re.search(r'\b' + re.escape(pattern) + r'\b', user_message):
                return entry
    # Default fallback if no match is found
    return {
        "intent": "default",
        "chatbot_response": "I’m here to help. Can you tell me more?",
        "emotion": "neutral",
        "suggested_action": "inquire"
    }

# Log user feedback for continuous learning
def log_feedback(user_message, response, rating):
    feedback_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_message": user_message,
        "chatbot_response": response,
        "user_feedback_rating": rating
    }
    with open("feedback_log.jsonl", "a") as f:
        f.write(json.dumps(feedback_entry) + "\n")

# Streamlit app
def main():
    st.set_page_config(page_title="Pandora - Mental Health Chatbot", layout="centered")
    
    # Styling for accessibility
    st.markdown(
        """
        <style>
        .stTextInput > div > div > input {
            font-size: 18px;
        }
        .stButton > button {
            font-size: 18px;
            background-color: #4CAF50;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Pandora - Your Mental Health Companion")
    st.write("Hello! I’m Pandora, here to support you with a caring and empathetic ear. How can I help you today?")

    # Initialize session state for conversation history
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "awaiting_feedback" not in st.session_state:
        st.session_state.awaiting_feedback = False
    if "last_response" not in st.session_state:
        st.session_state.last_response = ""

    # Load dataset
    dataset = load_dataset()

    # Display conversation history
    for message in st.session_state.conversation:
        if message["sender"] == "user":
            st.markdown(f"**You:** {message['text']}")
        else:
            st.markdown(f"**Pandora:** {message['text']}")

    # User input
    user_message = st.text_input("Type your message here:", key="user_input", placeholder="e.g., I feel sad today...")

    if st.button("Send") and user_message:
        # Add user message to conversation
        st.session_state.conversation.append({"sender": "user", "text": user_message})

        # Match intent and get response
        matched_entry = match_intent(user_message, dataset)
        response = matched_entry["chatbot_response"]

        # Add bot response to conversation
        st.session_state.conversation.append({"sender": "bot", "text": response})
        st.session_state.last_response = response
        st.session_state.awaiting_feedback = True

        # Refresh the page to show the new messages
        st.rerun()

    # Collect feedback if a response was just given
    if st.session_state.awaiting_feedback:
        st.write("How helpful was my response? (1 = Not helpful, 5 = Very helpful)")
        feedback = st.slider("Rate my response:", 1, 5, 3, key="feedback_slider")
        if st.button("Submit Feedback"):
            # Log feedback
            log_feedback(st.session_state.conversation[-2]["text"], st.session_state.last_response, feedback)
            st.session_state.awaiting_feedback = False
            st.write("Thank you for your feedback! How can I assist you next?")
            st.rerun()

if __name__ == "__main__":
    main()
