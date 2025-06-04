import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psutil
import gc
from datetime import datetime, timedelta
import json
import io
from functools import lru_cache
import hashlib
import time
from textblob import TextBlob
import pickle
import sqlite3
from passlib.hash import bcrypt
from transformers import pipeline
import plotly.figure_factory as ff
import os

# Page configuration
st.set_page_config(
    page_title="Mental Health Support Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .mood-card { /* ... existing styles ... */ }
    .stress-card { /* ... existing styles ... */ }
    .memory-card { /* ... existing styles ... */ }
    .chat-message { /* ... existing styles ... */ }
    .user-message { /* ... existing styles ... */ }
    .bot-message { /* ... existing styles ... */ }
    .crisis-alert { /* ... existing styles ... */ }
    .metric-container { /* ... existing styles ... */ }
    .trend-up { /* ... existing styles ... */ }
    .trend-down { /* ... existing styles ... */ }
    .trend-stable { /* ... existing styles ... */ }
    .auth-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Database setup for user authentication and data persistence
def init_db():
    conn = sqlite3.connect('mental_health.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                (username TEXT PRIMARY KEY, password TEXT, created_at TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS mood_data 
                (username TEXT, timestamp TIMESTAMP, mood INTEGER, stress INTEGER, category TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history 
                (username TEXT, timestamp TIMESTAMP, role TEXT, content TEXT)''')
    conn.commit()
    conn.close()

# Initialize advanced NLP model
@st.cache_resource
def init_nlp_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Initialize session state
def initialize_session_state():
    defaults = {
        'messages': [],
        'mood_data': pd.DataFrame(columns=['timestamp', 'mood', 'stress', 'category']),
        'conversation_count': 0,
        'user_name': '',
        'current_mood': 3,
        'crisis_detected': False,
        'last_mood_check': None,
        'cache_hits': 0,
        'memory_optimized': False,
        'authenticated': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Authentication functions
def register_user(username, password):
    conn = sqlite3.connect('mental_health.db')
    c = conn.cursor()
    try:
        hashed_password = bcrypt.hash(password)
        c.execute("INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)",
                 (username, hashed_password, datetime.now()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect('mental_health.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result and bcrypt.verify(password, result[0]):
        return True
    return False

def load_user_data(username):
    conn = sqlite3.connect('mental_health.db')
    mood_data = pd.read_sql_query(
        "SELECT timestamp, mood, stress, category FROM mood_data WHERE username = ?",
        conn, params=(username,))
    chat_history = pd.read_sql_query(
        "SELECT timestamp, role, content FROM chat_history WHERE username = ?",
        conn, params=(username,))
    conn.close()
    
    st.session_state.mood_data = mood_data
    st.session_state.messages = [
        {"role": row['role'], "content": row['content']}
        for _, row in chat_history.iterrows()
    ]

def save_user_data(username):
    conn = sqlite3.connect('mental_health.db')
    c = conn.cursor()
    
    # Save mood data
    for _, row in st.session_state.mood_data.iterrows():
        c.execute("INSERT OR REPLACE INTO mood_data VALUES (?, ?, ?, ?, ?)",
                 (username, row['timestamp'], row['mood'], row['stress'], row['category']))
    
    # Save chat history
    for msg in st.session_state.messages:
        c.execute("INSERT OR REPLACE INTO chat_history VALUES (?, ?, ?, ?)",
                 (username, datetime.now(), msg['role'], msg['content']))
    
    conn.commit()
    conn.close()

# Session backup/restore
def backup_session():
    backup_data = {
        'messages': st.session_state.messages,
        'mood_data': st.session_state.mood_data.to_dict(),
        'conversation_count': st.session_state.conversation_count,
        'current_mood': st.session_state.current_mood,
        'crisis_detected': st.session_state.crisis_detected
    }
    with open(f"backup_{st.session_state.user_name}.pkl", "wb") as f:
        pickle.dump(backup_data, f)
    return backup_data

def restore_session(username):
    try:
        with open(f"backup_{username}.pkl", "rb") as f:
            backup_data = pickle.load(f)
        st.session_state.messages = backup_data['messages']
        st.session_state.mood_data = pd.DataFrame(backup_data['mood_data'])
        st.session_state.conversation_count = backup_data['conversation_count']
        st.session_state.current_mood = backup_data['current_mood']
        st.session_state.crisis_detected = backup_data['crisis_detected']
        return True
    except FileNotFoundError:
        return False

# Enhanced NLP analysis
@st.cache_data(ttl=600, max_entries=50)
def analyze_mood_from_text(text, _nlp_model):
    if not text or len(text.strip()) < 3:
        return 3, 3
    
    # Use advanced NLP model
    result = _nlp_model(text)[0]
    sentiment_score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
    
    # Convert to mood scale (1-5)
    mood_score = max(1, min(5, int((sentiment_score + 1) * 2 + 1)))
    
    # Stress analysis with keywords
    stress_keywords = ['stress', 'anxious', 'worried', 'panic', 'overwhelm', 'pressure']
    crisis_keywords = ['suicide', 'kill myself', 'end it all', 'worthless', 'hopeless']
    
    text_lower = text.lower()
    stress_score = 3
    if any(word in text_lower for word in crisis_keywords):
        stress_score = 5
        mood_score = 1
    elif any(word in text_lower for word in stress_keywords):
        stress_score = 4
    elif sentiment_score < -0.5:
        stress_score = 4
    elif sentiment_score > 0.5:
        stress_score = 2
        
    return mood_score, stress_score

# Enhanced visualization
@st.cache_data(ttl=300)
def create_interactive_mood_chart(mood_data):
    if mood_data.empty:
        return None
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Mood Trends', 'Stress Levels', 'Mood vs Stress Correlation'),
        vertical_spacing=0.1
    )
    
    # Mood trend
    fig.add_trace(
        go.Scatter(
            x=mood_data['timestamp'],
            y=mood_data['mood'],
            mode='lines+markers',
            name='Mood',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Stress trend
    fig.add_trace(
        go.Scatter(
            x=mood_data['timestamp'],
            y=mood_data['stress'],
            mode='lines+markers',
            name='Stress',
            line=dict(color='#f093fb', width=3),
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    
    # Correlation plot
    fig.add_trace(
        go.Scatter(
            x=mood_data['mood'],
            y=mood_data['stress'],
            mode='markers',
            name='Correlation',
            marker=dict(size=10, color=mood_data['timestamp'].astype('int64')//10**9, colorscale='Viridis')
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Advanced Mood & Stress Analytics",
        title_x=0.5,
        hovermode='x unified'
    )
    
    return fig

# Additional mental health resources
MENTAL_HEALTH_RESOURCES = {
    'International': [
        {'name': 'WHO Mental Health', 'url': 'https://www.who.int/health-topics/mental-health'},
        {'name': 'MentalHealth.gov', 'url': 'https://www.mentalhealth.gov/'}
    ],
    'US': [
        {'name': 'NAMI Helpline', 'phone': '1-800-950-6264', 'url': 'https://www.nami.org/help'},
        {'name': '988 Suicide & Crisis Lifeline', 'phone': '988', 'url': 'https://988lifeline.org/'}
    ],
    'UK': [
        {'name': 'Samaritans', 'phone': '116 123', 'url': 'https://www.samaritans.org/'},
        {'name': 'Mind', 'phone': '0300 123 3393', 'url': 'https://www.mind.org.uk/'}
    ]
}

# Other existing functions (get_system_memory_info, hash_text, calculate_trends, create_category_chart,
# optimize_memory, categorize_conversation, detect_crisis, get_ai_response, log_mood_data, export_data_as_csv)
# remain unchanged from the original implementation

def main():
    init_db()
    initialize_session_state()
    nlp_model = init_nlp_model()

    # Authentication
    if not st.session_state.authenticated:
        with st.container():
            st.markdown('<div class="auth-container">', unsafe_allow_html=True)
            st.subheader("🔐 User Authentication")
            
            auth_choice = st.radio("Login or Register", ["Login", "Register"])
            
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if auth_choice == "Register":
                if st.button("Register"):
                    if register_user(username, password):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Username already exists!")
            else:
                if st.button("Login"):
                    if authenticate_user(username, password):
                        st.session_state.authenticated = True
                        st.session_state.user_name = username
                        load_user_data(username)
                        if restore_session(username):
                            st.success("Session restored successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials!")
            st.markdown('</div>', unsafe_allow_html=True)
        return

    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>🧠 Mental Health Support Chatbot</h1>
        <p>Welcome, {st.session_state.user_name}! Your AI companion for mental wellness</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Analytics Dashboard")
        
        # Memory info
        memory_info = get_system_memory_info()
        st.markdown(f"""
        <div class="memory-card">
            <h4>💾 System Memory</h4>
            <p>Used: {memory_info['used']} GB / {memory_info['total']} GB</p>
            <p>Usage: {memory_info['percent']:.1f}%</p>
            <p>Cache Hits: {st.session_state.cache_hits}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mood statistics and other sidebar components remain the same
        
        # Additional resources
        st.subheader("📚 Mental Health Resources")
        for region, resources in MENTAL_HEALTH_RESOURCES.items():
            with st.expander(region):
                for resource in resources:
                    st.markdown(f"[{resource['name']}]({resource.get('url', '#')})")
                    if 'phone' in resource:
                        st.write(f"📞 {resource['phone']}")
        
        # Session backup/restore
        st.subheader("💾 Session Management")
        if st.button("Backup Session"):
            backup_session()
            st.success("Session backed up successfully!")
        
        if st.button("Restore Session"):
            if restore_session(st.session_state.user_name):
                st.success("Session restored successfully!")
                st.rerun()
            else:
                st.error("No backup found!")
        
        # Logout
        if st.button("Logout"):
            save_user_data(st.session_state.user_name)
            st.session_state.authenticated = False
            st.session_state.user_name = ''
            st.rerun()

    # Main chat interface (remains mostly the same, with updated mood analysis)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>{st.session_state.user_name}:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>AI Assistant:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        if st.session_state.crisis_detected:
            st.markdown("""
            <div class="crisis-alert">
                <h3>🚨 Crisis Support Resources Available</h3>
                <p>If you're having thoughts of self-harm, please reach out for immediate help.</p>
            </div>
            """, unsafe_allow_html=True)
        
        user_input = st.chat_input("Share what's on your mind...")
        
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            mood_score, stress_score = analyze_mood_from_text(user_input, nlp_model)
            category = categorize_conversation(user_input)
            crisis = detect_crisis(user_input)
            
            if crisis:
                st.session_state.crisis_detected = True
            
            log_mood_data(mood_score, stress_score, category)
            save_user_data(st.session_state.user_name)
            
            ai_response = get_ai_response(user_input, mood_score, stress_score, category)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            
            st.session_state.conversation_count += 1
            st.rerun()
    
    with col2:
        st.subheader("📈 Advanced Analytics")
        
        if not st.session_state.mood_data.empty:
            mood_chart = create_interactive_mood_chart(st.session_state.mood_data)
            if mood_chart:
                st.plotly_chart(mood_chart, use_container_width=True)
            
            category_chart = create_category_chart(st.session_state.mood_data)
            if category_chart:
                st.plotly_chart(category_chart, use_container_width=True)
            
            st.subheader("📋 Recent Mood Entries")
            recent_entries = st.session_state.mood_data.tail(5)
            for _, entry in recent_entries.iterrows():
                mood_emoji = "😊" if entry['mood'] >= 4 else "😐" if entry['mood'] >= 3 else "😔"
                stress_emoji = "😌" if entry['stress'] <= 2 else "😰" if entry['stress'] >= 4 else "🤔"
                st.write(f"{mood_emoji} {stress_emoji} {entry['timestamp'].strftime('%H:%M')} - {entry['category']}")
        
        else:
            st.info("Start chatting to see analytics and mood tracking data!")

if __name__ == "__main__":
    main()
