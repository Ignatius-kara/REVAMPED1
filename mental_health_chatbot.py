```python
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
from contextlib import contextmanager
try:
    from passlib.hash import bcrypt
except ImportError:
    st.error("passlib module not found. Please ensure it is installed.")
    st.stop()
try:
    from transformers import pipeline
except ImportError:
    st.warning("Transformers not available; using TextBlob for sentiment analysis.")
    pipeline = None
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Mental Health Support Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (unchanged)
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
    .mood-card { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white; margin: 0.5rem 0; }
    .stress-card { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white; margin: 0.5rem 0; }
    .memory-card { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white; margin: 0.5rem 0; }
    .chat-message { padding: 1rem; border-radius: 10px; margin: 0.5rem 0; animation: fadeIn 0.5s; }
    .user-message { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; }
    .bot-message { background: #f0f2f6; color: #262730; border-left: 4px solid #667eea; }
    .crisis-alert { background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); border: 2px solid #ff6b6b; padding: 1rem; border-radius: 10px; text-align: center; margin: 1rem 0; }
    @keyframes fadeIn { from {opacity: 0; transform: translateY(10px);} to {opacity: 1; transform: translateY(0);} }
    .metric-container { display: flex; justify-content: space-around; margin: 1rem 0; }
    .trend-up { color: #28a745; }
    .trend-down { color: #dc3545; }
    .trend-stable { color: #ffc107; }
    .auth-container { background: #f0f2f6; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# Default dataset if file is missing
DEFAULT_DATASET = [
    {"user": "I'm feeling anxious", "response": "I'm here to help. Try a deep breathing exercise: inhale for 4, hold for 4, exhale for 6."},
    {"user": "I'm sad today", "response": "I'm sorry to hear that. Would you like to share more about what's going on?"}
]

# Load dataset with fallback
@st.cache_data
def load_dataset():
    try:
        dataset = []
        with open("optimized_mental_health_chatbot_dataset.jsonl", "r") as f:
            for line in f:
                dataset.append(json.loads(line))
        logger.info("Dataset loaded successfully")
        return dataset
    except FileNotFoundError:
        logger.warning("Dataset file not found; using default dataset")
        return DEFAULT_DATASET
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return DEFAULT_DATASET

# Database connection pooling
@contextmanager
def get_db_connection():
    conn = sqlite3.connect('mental_health.db', check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users 
                    (username TEXT PRIMARY KEY, password TEXT, created_at TIMESTAMP)''')
        c.execute('''CREATE TABLE IF NOT EXISTS mood_data 
                    (username TEXT, timestamp TIMESTAMP, mood INTEGER, stress INTEGER, category TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS chat_history 
                    (username TEXT, timestamp TIMESTAMP, role TEXT, content TEXT)''')
        conn.commit()

# Initialize lightweight NLP model for deployment
@st.cache_resource
def init_nlp_model():
    if pipeline is None:
        logger.warning("Transformers not available; using TextBlob fallback")
        return None
    try:
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        logger.error(f"Failed to load NLP model: {e}")
        return None

# Initialize session state
def initialize_session_state():
    defaults = {
        'messages': [],
        'mood_data': pd.DataFrame(columns=['timestamp', 'mood', 'stress', 'category'], dtype=object),
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
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            hashed_password = bcrypt.hash(password)
            c.execute("INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)",
                     (username, hashed_password, datetime.now()))
            conn.commit()
            return True
    except sqlite3.IntegrityError:
        return False
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return False

def authenticate_user(username, password):
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT password FROM users WHERE username = ?", (username,))
            result = c.fetchone()
            if result and bcrypt.verify(password, result[0]):
                return True
            return False
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return False

def load_user_data(username):
    try:
        with get_db_connection() as conn:
            mood_data = pd.read_sql_query(
                "SELECT timestamp, mood, stress, category FROM mood_data WHERE username = ? ORDER BY timestamp DESC LIMIT 100",
                conn, params=(username,))
            chat_history = pd.read_sql_query(
                "SELECT timestamp, role, content FROM chat_history WHERE username = ? ORDER BY timestamp DESC LIMIT 50",
                conn, params=(username,))
        
        st.session_state.mood_data = mood_data
        st.session_state.messages = [
            {"role": row['role'], "content": row['content']}
            for _, row in chat_history.iterrows()
        ]
    except Exception as e:
        logger.error(f"Error loading user data: {e}")

def save_user_data(username):
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            mood_data_tuples = [
                (username, row['timestamp'], row['mood'], row['stress'], row['category'])
                for _, row in st.session_state.mood_data.tail(10).iterrows()
            ]
            c.executemany("INSERT OR REPLACE INTO mood_data VALUES (?, ?, ?, ?, ?)", mood_data_tuples)
            
            chat_tuples = [
                (username, datetime.now(), msg['role'], msg['content'])
                for msg in st.session_state.messages[-10:]
            ]
            c.executemany("INSERT OR REPLACE INTO chat_history VALUES (?, ?, ?, ?)", chat_tuples)
            
            conn.commit()
    except Exception as e:
        logger.error(f"Error saving user data: {e}")

# Session backup/restore
def backup_session():
    try:
        backup_data = {
            'messages': st.session_state.messages[-50:],
            'mood_data': st.session_state.mood_data.tail(100).to_dict(),
            'conversation_count': st.session_state.conversation_count,
            'current_mood': st.session_state.current_mood,
            'crisis_detected': st.session_state.crisis_detected
        }
        backup_file = f"backup_{st.session_state.user_name}.pkl"
        with open(backup_file, "wb") as f:
            pickle.dump(backup_data, f)
        return backup_data
    except Exception as e:
        logger.error(f"Backup error: {e}")
        return None

def restore_session(username):
    try:
        backup_file = f"backup_{username}.pkl"
        with open(backup_file, "rb") as f:
            backup_data = pickle.load(f)
        st.session_state.messages = backup_data['messages']
        st.session_state.mood_data = pd.DataFrame(backup_data['mood_data'])
        st.session_state.conversation_count = backup_data['conversation_count']
        st.session_state.current_mood = backup_data['current_mood']
        st.session_state.crisis_detected = backup_data['crisis_detected']
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        logger.error(f"Restore error: {e}")
        return False

# Optimized NLP analysis
@st.cache_data(ttl=300, max_entries=50)
def analyze_mood_from_text(text, _nlp_model):
    if not text or len(text.strip()) < 3:
        return 3, 3
    
    try:
        if _nlp_model:
            result = _nlp_model(text)[0]
            sentiment_score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
        else:
            blob = TextBlob(text.lower())
            sentiment_score = blob.sentiment.polarity
        
        mood_score = max(1, min(5, int((sentiment_score + 1) * 2 + 1)))
        
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
    except Exception as e:
        logger.error(f"Mood analysis error: {e}")
        return 3, 3

# Optimized visualization
@st.cache_data(ttl=300)
def create_interactive_mood_chart(mood_data):
    if mood_data.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Mood Trends', 'Stress Levels'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(
            x=mood_data['timestamp'],
            y=mood_data['mood'],
            mode='lines+markers',
            name='Mood',
            line=dict(color='#667eea', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=mood_data['timestamp'],
            y=mood_data['stress'],
            mode='lines+markers',
            name='Stress',
            line=dict(color='#f093fb', width=2),
            marker=dict(size=6)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Mood & Stress Analytics",
        title_x=0.5,
        hovermode='x'
    )
    
    return fig

# System memory info
@st.cache_data(ttl=300, max_entries=50)
def get_system_memory_info():
    try:
        memory = psutil.virtual_memory()
        return {
            'total': round(memory.total / (1024**3), 2),
            'available': round(memory.available / (1024**3), 2),
            'used': round(memory.used / (1024**3), 2),
            'percent': memory.percent
        }
    except Exception as e:
        logger.error(f"Memory info error: {e}")
        return {'total': 0, 'available': 0, 'used': 0, 'percent': 0}

@lru_cache(maxsize=128)
def hash_text(text):
    return hashlib.md5(text.encode()).hexdigest()

@st.cache_data(ttl=300)
def calculate_trends(mood_data):
    if len(mood_data) < 2:
        return {"mood_trend": "stable", "stress_trend": "stable", "mood_change": 0, "stress_change": 0}
    
    recent_data = mood_data.tail(5)
    older_data = mood_data.head(max(1, len(mood_data) - 5))
    
    recent_mood_avg = recent_data['mood'].mean()
    recent_stress_avg = recent_data['stress'].mean()
    older_mood_avg = older_data['mood'].mean()
    older_stress_avg = older_data['stress'].mean()
    
    mood_change = recent_mood_avg - older_mood_avg
    stress_change = recent_stress_avg - older_stress_avg
    
    mood_trend = "improving" if mood_change > 0.2 else "declining" if mood_change < -0.2 else "stable"
    stress_trend = "improving" if stress_change < -0.2 else "worsening" if stress_change > 0.2 else "stable"
    
    return {
        "mood_trend": mood_trend,
        "stress_trend": stress_trend,
        "mood_change": round(mood_change, 2),
        "stress_change": round(stress_change, 2)
    }

@st.cache_data(ttl=600)
def create_category_chart(mood_data):
    if mood_data.empty:
        return None
    
    category_counts = mood_data['category'].value_counts()
    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title="Conversation Topics",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=300)
    return fig

def optimize_memory():
    if not st.session_state.get('memory_optimized', False):
        if len(st.session_state.mood_data) > 50:
            st.session_state.mood_data = st.session_state.mood_data.tail(50).reset_index(drop=True)
        if len(st.session_state.messages) > 30:
            st.session_state.messages = st.session_state.messages[-30:]
        gc.collect()
        memory_info = get_system_memory_info()
        if memory_info['percent'] > 75:
            st.cache_data.clear()
            st.session_state.cache_hits = 0
        st.session_state.memory_optimized = True
        logger.info("Memory optimized")

def categorize_conversation(message):
    categories = {
        'anxiety': ['anxious', 'worry', 'nervous', 'panic', 'fear'],
        'depression': ['sad', 'depressed', 'hopeless', 'empty', 'worthless'],
        'stress': ['stress', 'pressure', 'overwhelm', 'burden', 'exhausted'],
        'relationships': ['relationship', 'family', 'friends', 'partner', 'social'],
        'work': ['work', 'job', 'career', 'boss', 'colleague'],
        'general': []
    }
    message_lower = message.lower()
    for category, keywords in categories.items():
        if any(keyword in message_lower for keyword in keywords):
            return category
    return 'general'

def detect_crisis(message):
    crisis_keywords = [
        'suicide', 'kill myself', 'end it all', 'not worth living',
        'better off dead', 'want to die', 'end my life', 'hurt myself'
    ]
    return any(keyword in message.lower() for keyword in crisis_keywords)

@lru_cache(maxsize=128)
def get_ai_response(message, mood_score, stress_score, category):
    st.session_state.cache_hits += 1
    base_responses = {
        'anxiety': ["Let's try a quick breathing exercise: inhale for 4, hold for 4, exhale for 6."],
        'depression': ["Your feelings are valid. Would you like to share more about what's been going on?"],
        'stress': ["Stress can be tough. What's been the main source of stress for you?"],
        'relationships': ["Relationships can be complex. Would you like to talk about what's happening?"],
        'work': ["Work stress is common. Have you been able to find any strategies that help?"]
    }
    
    crisis_response = """🚨 **CRISIS SUPPORT** 🚨
    I'm concerned about what you've shared. Please reach out for help:
    • National Suicide Prevention Lifeline: 988
    • Crisis Text Line: Text HOME TO 741741
    • Emergency Services: 911"""
    
    if detect_crisis(message):
        return crisis_response
    
    responses = base_responses.get(category, ["I'm here to listen. What's on your mind?"])
    return responses[0]

def log_mood_data(mood, stress, category):
    new_entry = pd.DataFrame({
        'timestamp': [datetime.now()],
        'mood': [mood],
        'stress': [stress],
        'category': [category]
    }, dtype=str)  # Explicitly set dtype to avoid type mismatches
    if st.session_state.mood_data.empty:
        st.session_state.mood_data = new_entry
    else:
        st.session_state.mood_data = pd.concat([st.session_state.mood_data, new_entry], ignore_index=True)
    if len(st.session_state.mood_data) % 5 == 0:
        optimize_memory()

def export_data_as_csv(data, filename_prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue(), filename

# Mental health resources
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

def main():
    init_db()
    initialize_session_state()
    nlp_model = init_nlp_model()
    dataset = load_dataset()  # Load dataset with fallback

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
                            st.success("Session restored!")
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
        memory_info = get_system_memory_info()
        st.markdown(f"""
        <div class="memory-card">
            <h4>💾 System Memory</h4>
            <p>Used: {memory_info['used']} GB / {memory_info['total']} GB</p>
            <p>Usage: {memory_info['percent']:.1f}%</p>
            <p>Cache Hits: {st.session_state.cache_hits}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.mood_data.empty:
            avg_mood = st.session_state.mood_data['mood'].mean()
            avg_stress = st.session_state.mood_data['stress'].mean()
            trends = calculate_trends(st.session_state.mood_data)
            
            mood_emoji = "😊" if avg_mood >= 4 else "😐" if avg_mood >= 3 else "😔"
            stress_emoji = "😌" if avg_stress <= 2 else "😰" if avg_stress >= 4 else "🤔"
            trend_mood_icon = "📈" if trends['mood_trend'] == 'improving' else "📉" if trends['mood_trend'] == 'declining' else "➡️"
            trend_stress_icon = "📉" if trends['stress_trend'] == 'improving' else "📈" if trends['stress_trend'] == 'worsening' else "➡️"
            
            st.markdown(f"""
            <div class="mood-card">
                <h4>{mood_emoji} Average Mood</h4>
                <h2>{avg_mood:.1f}/5</h2>
                <p>{trend_mood_icon} {trends['mood_trend'].title()}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stress-card">
                <h4>{stress_emoji} Average Stress</h4>
                <h2>{avg_stress:.1f}/5</h2>
                <p>{trend_stress_icon} {trends['stress_trend'].title()}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("💬 Total Interactions", len(st.session_state.mood_data))
        
        st.subheader("🎭 Quick Mood Check")
        mood_input = st.selectbox("How are you feeling?", [1, 2, 3, 4, 5], index=2,
                                 format_func=lambda x: f"{x} - {'😢' if x<=2 else '😐' if x==3 else '😊' if x<=4 else '😄'}")
        stress_input = st.selectbox("Stress level?", [1, 2, 3, 4, 5], index=2,
                                   format_func=lambda x: f"{x} - {'😌' if x<=2 else '🤔' if x==3 else '😰' if x<=4 else '🤯'}")
        
        if st.button("📝 Log Mood"):
            log_mood_data(mood_input, stress_input, 'manual_entry')
            save_user_data(st.session_state.user_name)
            st.success("Mood logged!")
            st.rerun()
        
        st.subheader("📤 Export Data")
        if not st.session_state.mood_data.empty:
            mood_csv, mood_filename = export_data_as_csv(st.session_state.mood_data, "mood_data")
            st.download_button(
                label="📊 Download Mood Data",
                data=mood_csv,
                file_name=mood_filename,
                mime="text/csv"
            )
        
        if st.session_state.messages:
            chat_df = pd.DataFrame([
                {
                    'timestamp': datetime.now() - timedelta(minutes=len(st.session_state.messages)-i),
                    'role': msg['role'],
                    'content': msg['content']
                }
                for i, msg in enumerate(st.session_state.messages)
            ])
            chat_csv, chat_filename = export_data_as_csv(chat_df, "chat_history")
            st.download_button(
                label="💬 Download Chat History",
                data=chat_csv,
                file_name=chat_filename,
                mime="text/csv"
            )
        
        st.subheader("📚 Mental Health Resources")
        for region, resources in MENTAL_HEALTH_RESOURCES.items():
            with st.expander(region):
                for resource in resources:
                    st.markdown(f"[{resource['name']}]({resource.get('url', '#')})")
                    if 'phone' in resource:
                        st.write(f"📞 {resource['phone']}")
        
        st.subheader("💾 Session Management")
        if st.button("Backup Session"):
            if backup_session():
                st.success("Session backed up!")
        
        if st.button("Restore Session"):
            if restore_session(st.session_state.user_name):
                st.success("Session restored!")
                st.rerun()
            else:
                st.error("No backup found!")
        
        if st.button("🧹 Optimize Memory"):
            optimize_memory()
            st.success("Memory optimized!")
            st.rerun()
        
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.session_state.cache_hits = 0
            st.success("Cache cleared!")
            st.rerun()
        
        if st.button("Logout"):
            save_user_data(st.session_state.user_name)
            st.session_state.authenticated = False
            st.session_state.user_name = ''
            st.rerun()

    # Main chat interface
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
                <h3>🚨 Crisis Support Resources</h3>
                <p>Please reach out for immediate help.</p>
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
        st.subheader("📈 Analytics")
        if not st.session_state.mood_data.empty:
            mood_chart = create_interactive_mood_chart(st.session_state.mood_data)
            if mood_chart:
                st.plotly_chart(mood_chart, use_container_width=True)
            
            category_chart = create_category_chart(st.session_state.mood_data)
            if category_chart:
                st.plotly_chart(category_chart, use_container_width=True)
            
            st.subheader("📋 Recent Mood Entries")
            recent_entries = st.session_state.mood_data.tail(3)
            for _, entry in recent_entries.iterrows():
                mood_emoji = "😊" if float(entry['mood']) >= 4 else "😐" if float(entry['mood']) >= 3 else "😔"
                stress_emoji = "😌" if float(entry['stress']) <= 2 else "😰" if float(entry['stress']) >= 4 else "🤔"
                st.write(f"{mood_emoji} {stress_emoji} {entry['timestamp'].strftime('%H:%M')} - {entry['category']}")
        else:
            st.info("Start chatting to see analytics!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An error occurred. Please try again or contact support.")
```
