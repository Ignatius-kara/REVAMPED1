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
import nltk
import os

# Download NLTK data on first run
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown', quiet=True)

# Page configuration
st.set_page_config(
    page_title="Mental Health Support Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved styling
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
    .mood-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem 0;
    }
    .stress-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem 0;
    }
    .memory-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        animation: fadeIn 0.5s;
    }
    .user-message {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .bot-message {
        background: #f0f2f6;
        color: #262730;
        border-left: 4px solid #667eea;
    }
    .crisis-alert {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        border: 2px solid #ff6b6b;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .trend-up { color: #28a745; }
    .trend-down { color: #dc3545; }
    .trend-stable { color: #ffc107; }
</style>
""", unsafe_allow_html=True)

# Initialize session state with caching optimization
def initialize_session_state():
    """Initialize session state with optimized default values"""
    defaults = {
        'messages': [],
        'mood_data': pd.DataFrame(columns=['timestamp', 'mood', 'stress', 'category']),
        'conversation_count': 0,
        'user_name': '',
        'current_mood': 3,
        'crisis_detected': False,
        'last_mood_check': None,
        'cache_hits': 0,
        'memory_optimized': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Memory management and caching functions
@st.cache_data(ttl=300, max_entries=100)  # Cache for 5 minutes, max 100 entries
def get_system_memory_info():
    """Get system memory information with caching"""
    try:
        memory = psutil.virtual_memory()
        return {
            'total': round(memory.total / (1024**3), 2),
            'available': round(memory.available / (1024**3), 2),
            'used': round(memory.used / (1024**3), 2),
            'percent': memory.percent
        }
    except Exception:
        return {
            'total': 0,
            'available': 0,
            'used': 0,
            'percent': 0
        }

@lru_cache(maxsize=128)
def hash_text(text):
    """Create hash for text caching"""
    return hashlib.md5(text.encode()).hexdigest()

@st.cache_data(ttl=600, max_entries=50)  # Cache for 10 minutes
def analyze_mood_from_text(text):
    """Analyze mood from text with caching"""
    if not text or len(text.strip()) < 3:
        return 3, 3  # neutral defaults
    
    try:
        # Use TextBlob for sentiment analysis
        blob = TextBlob(text.lower())
        sentiment = blob.sentiment
        
        # Convert polarity (-1 to 1) to mood scale (1-5)
        mood_score = max(1, min(5, int((sentiment.polarity + 1) * 2 + 1)))
        
        # Stress indicators
        stress_keywords = ['stress', 'anxious', 'worried', 'panic', 'overwhelm', 'pressure']
        crisis_keywords = ['suicide', 'kill myself', 'end it all', 'worthless', 'hopeless']
        
        text_lower = text.lower()
        stress_score = 3  # default
        
        if any(word in text_lower for word in crisis_keywords):
            stress_score = 5
            mood_score = 1
        elif any(word in text_lower for word in stress_keywords):
            stress_score = 4
        elif sentiment.polarity < -0.3:
            stress_score = 4
        elif sentiment.polarity > 0.3:
            stress_score = 2
            
        return mood_score, stress_score
        
    except Exception:
        return 3, 3

@st.cache_data(ttl=300)
def calculate_trends(mood_data):
    """Calculate mood and stress trends with caching"""
    if len(mood_data) < 2:
        return {"mood_trend": "stable", "stress_trend": "stable", "mood_change": 0, "stress_change": 0}
    
    recent_data = mood_data.tail(10)  # Last 10 entries
    older_data = mood_data.head(max(1, len(mood_data) - 10))
    
    if len(older_data) == 0:
        return {"mood_trend": "stable", "stress_trend": "stable", "mood_change": 0, "stress_change": 0}
    
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

@st.cache_data(ttl=300, max_entries=20)
def create_mood_chart(mood_data):
    """Create mood trend chart with caching"""
    if mood_data.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Mood Trends', 'Stress Levels'),
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
    
    fig.update_layout(
        height=500,
        showlegend=True,
        title_text="Mood & Stress Analytics",
        title_x=0.5
    )
    
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Mood (1-5)", row=1, col=1)
    fig.update_yaxes(title_text="Stress (1-5)", row=2, col=1)
    
    return fig

@st.cache_data(ttl=600)
def create_category_chart(mood_data):
    """Create category distribution chart with caching"""
    if mood_data.empty:
        return None
    
    category_counts = mood_data['category'].value_counts()
    
    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title="Conversation Topics Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def optimize_memory():
    """Optimize memory usage with enhanced techniques"""
    if not st.session_state.get('memory_optimized', False):
        # Limit mood data to last 100 entries
        if len(st.session_state.mood_data) > 100:
            st.session_state.mood_data = st.session_state.mood_data.tail(100).reset_index(drop=True)
        
        # Limit messages to last 50
        if len(st.session_state.messages) > 50:
            st.session_state.messages = st.session_state.messages[-50:]
        
        # Force garbage collection
        gc.collect()
        
        # Clear specific caches if memory usage is high
        memory_info = get_system_memory_info()
        if memory_info['percent'] > 80:
            st.cache_data.clear()
            st.session_state.cache_hits = 0
        
        st.session_state.memory_optimized = True

def categorize_conversation(message):
    """Categorize conversation type with caching"""
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
    """Enhanced crisis detection with caching"""
    crisis_keywords = [
        'suicide', 'kill myself', 'end it all', 'not worth living',
        'better off dead', 'want to die', 'end my life', 'hurt myself',
        'self harm', 'cut myself', 'overdose', 'jumping'
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in crisis_keywords)

@lru_cache(maxsize=256)
def get_ai_response(message, mood_score, stress_score, category):
    """Generate AI response with caching based on input parameters"""
    st.session_state.cache_hits += 1
    
    base_responses = {
        'anxiety': [
            "I understand you're feeling anxious. Let's try some breathing exercises together. Take a deep breath in for 4 counts, hold for 4, and exhale for 6.",
            "Anxiety can feel overwhelming, but remember that these feelings are temporary. What's one small thing that usually helps you feel calmer?",
            "When anxiety strikes, grounding techniques can help. Can you name 5 things you can see around you right now?"
        ],
        'depression': [
            "I hear that you're going through a difficult time. Your feelings are valid, and it's okay to not be okay sometimes.",
            "Depression can make everything feel harder. Have you been able to do any small self-care activities today?",
            "Remember that seeking help is a sign of strength, not weakness. You don't have to go through this alone."
        ],
        'stress': [
            "Stress can be really challenging to manage. What's been the biggest source of stress for you lately?",
            "Let's work on breaking down what's stressing you into smaller, manageable pieces. What feels most urgent right now?",
            "Stress affects everyone differently. Have you tried any relaxation techniques that work for you?"
        ],
        'relationships': [
            "Relationships can be complex and emotionally challenging. It sounds like you're dealing with some difficult dynamics.",
            "Communication is key in relationships. Have you been able to express how you're feeling to the people involved?",
            "Setting healthy boundaries is important for your mental health. What boundaries do you think might help in this situation?"
        ],
        'work': [
            "Work-related stress is very common. It's important to find ways to manage it before it affects your overall well-being.",
            "Workplace challenges can impact our mental health significantly. Have you considered speaking with HR or a supervisor about your concerns?",
            "Work-life balance is crucial. What's one thing you could do today to create better boundaries between work and personal time?"
        ]
    }
    
    crisis_response = """🚨 **CRISIS SUPPORT RESOURCES** 🚨
    
I'm very concerned about what you've shared. Please know that you matter and there are people who want to help:

**Immediate Help:**
• **National Suicide Prevention Lifeline: 988**
• **Crisis Text Line: Text HOME to 741741**
• **Emergency Services: 911**

**Online Resources:**
• **SAMHSA: 1-800-662-4357**
• **Crisis Chat: suicidepreventionlifeline.org**

You don't have to go through this alone. Professional counselors are available 24/7 and want to help you through this difficult time."""

    if detect_crisis(message):
        return crisis_response
    
    # Adjust response based on mood and stress
    responses = base_responses.get(category, [
        "Thank you for sharing that with me. I'm here to listen and support you.",
        "It sounds like you're dealing with something important. Would you like to tell me more about how you're feeling?",
        "I appreciate you opening up. What would be most helpful for you right now?"
    ])
    
    # Select response based on mood/stress levels
    if mood_score <= 2 or stress_score >= 4:
        response_idx = 0  # More supportive response
    elif mood_score >= 4 and stress_score <= 2:
        response_idx = min(2, len(responses) - 1)  # More encouraging response
    else:
        response_idx = 1  # Balanced response
    
    return responses[response_idx]

def log_mood_data(mood, stress, category):
    """Log mood data with memory optimization"""
    new_entry = pd.DataFrame({
        'timestamp': [datetime.now()],
        'mood': [mood],
        'stress': [stress],
        'category': [category]
    })
    
    st.session_state.mood_data = pd.concat([st.session_state.mood_data, new_entry], ignore_index=True)
    
    # Auto-optimize memory every 10 entries
    if len(st.session_state.mood_data) % 10 == 0:
        optimize_memory()

def export_data_as_csv(data, filename_prefix):
    """Export data as CSV with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    
    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue(), filename

# Main application
def main():
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Mental Health Support Chatbot</h1>
        <p>Your AI companion for mental wellness with advanced analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with enhanced features
    with st.sidebar:
        st.header("📊 Analytics Dashboard")
        
        # Memory and performance info
        memory_info = get_system_memory_info()
        st.markdown(f"""
        <div class="memory-card">
            <h4>💾 System Memory</h4>
            <p>Used: {memory_info['used']} GB / {memory_info['total']} GB</p>
            <p>Usage: {memory_info['percent']:.1f}%</p>
            <p>Cache Hits: {st.session_state.cache_hits}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mood statistics
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
            
            # Total interactions
            st.metric("💬 Total Interactions", len(st.session_state.mood_data))
            
        # Quick mood check
        st.subheader("🎭 Quick Mood Check")
        mood_input = st.selectbox("How are you feeling?", [1, 2, 3, 4, 5], index=2,
                                 format_func=lambda x: f"{x} - {'😢' if x<=2 else '😐' if x==3 else '😊' if x<=4 else '😄'}")
        stress_input = st.selectbox("Stress level?", [1, 2, 3, 4, 5], index=2,
                                   format_func=lambda x: f"{x} - {'😌' if x<=2 else '🤔' if x==3 else '😰' if x<=4 else '🤯'}")
        
        if st.button("📝 Log Mood"):
            log_mood_data(mood_input, stress_input, 'manual_entry')
            st.success("Mood logged successfully!")
            st.rerun()
        
        # Export options
        st.subheader("📤 Export Data")
        
        if not st.session_state.mood_data.empty:
            # Export mood data
            mood_csv, mood_filename = export_data_as_csv(st.session_state.mood_data, "mood_data")
            st.download_button(
                label="📊 Download Mood Data",
                data=mood_csv,
                file_name=mood_filename,
                mime="text/csv"
            )
        
        if st.session_state.messages:
            # Export chat history
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
        
        # Memory optimization button
        if st.button("🧹 Optimize Memory"):
            optimize_memory()
            st.success("Memory optimized!")
            st.rerun()
        
        # Clear cache button
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.session_state.cache_hits = 0
            st.success("Cache cleared!")
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>AI Assistant:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Crisis alert
        if st.session_state.crisis_detected:
            st.markdown("""
            <div class="crisis-alert">
                <h3>🚨 Crisis Support Resources Available</h3>
                <p>If you're having thoughts of self-harm, please reach out for immediate help.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Chat input
        user_input = st.chat_input("Share what's on your mind...")
        
        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Analyze mood and detect crisis
            mood_score, stress_score = analyze_mood_from_text(user_input)
            category = categorize_conversation(user_input)
            crisis = detect_crisis(user_input)
            
            if crisis:
                st.session_state.crisis_detected = True
            
            # Log mood data
            log_mood_data(mood_score, stress_score, category)
            
            # Generate AI response with caching
            ai_response = get_ai_response(user_input, mood_score, stress_score, category)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            
            st.session_state.conversation_count += 1
            st.rerun()
    
    with col2:
        st.subheader("📈 Analytics Visualizations")
        
        if not st.session_state.mood_data.empty:
            # Mood trend chart
            mood_chart = create_mood_chart(st.session_state.mood_data)
            if mood_chart:
                st.plotly_chart(mood_chart, use_container_width=True)
            
            # Category distribution
            if len(st.session_state.mood_data) > 1:
                category_chart = create_category_chart(st.session_state.mood_data)
                if category_chart:
                    st.plotly_chart(category_chart, use_container_width=True)
            
            # Recent mood entries
            st.subheader("📋 Recent Mood Entries")
            recent_entries = st.session_state.mood_data.tail(5)
            for _, entry in recent_entries.iterrows():
                mood_emoji = "😊" if entry['mood'] >= 4 else "😐" if entry['mood'] >= 3 else "😔"
                stress_emoji = "😌" if entry['stress'] <= 2 else "😰" if entry['stress'] >= 4 else "🤔"
                st.write(f"{mood_emoji} {stress_emoji} {entry['timestamp'].strftime('%H:%M')} - {entry['category']}")
        
        else:
            st.info("Start chatting to see analytics and mood tracking data!")
    
    # Footer
    st.markdown("""
    ---
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🧠 Mental Health Support Chatbot with Advanced Analytics</p>
        <p>Remember: This AI assistant is not a replacement for professional mental health care.</p>
        <p>💾 Memory optimized • 📊 Real-time analytics • 🗂️ Data export capabilities</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
