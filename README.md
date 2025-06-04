# 🧠 Mental Health Support Chatbot

A comprehensive AI-powered mental health support chatbot with advanced analytics, mood tracking, and real-time visualizations. Built with Streamlit and designed for deployment on Vercel.

## ✨ Features

### 🎯 Core Functionality
- **AI-Powered Conversations**: Intelligent responses based on conversation context and mood
- **Crisis Detection**: Automatic detection of crisis situations with immediate resource links
- **Multi-Category Support**: Specialized responses for anxiety, depression, stress, relationships, and work issues

### 📊 Advanced Analytics
- **Mood Tracking**: Automatic mood analysis from conversation text
- **Stress Monitoring**: Real-time stress level assessment
- **Trend Analysis**: Visual representation of mood and stress patterns over time
- **Category Analytics**: Distribution analysis of conversation topics

### 💾 Data Management
- **CSV Export**: Download mood data and chat history
- **Memory Optimization**: Intelligent memory management with automatic cleanup
- **Data Persistence**: Session-based data storage with export capabilities

### 🚀 Performance Features
- **Smart Caching**: Multi-layer caching system for optimal performance
- **Memory Monitoring**: Real-time system resource tracking
- **LRU Cache**: Efficient response caching with Least Recently Used algorithm
- **Auto-Optimization**: Automatic memory cleanup and data trimming

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly
- **Natural Language Processing**: TextBlob, NLTK
- **System Monitoring**: psutil
- **Deployment**: Vercel

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd mental-health-chatbot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser**
Navigate to `http://localhost:8501`

### Vercel Deployment

1. **Push to GitHub**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy on Vercel**
- Visit [vercel.com](https://vercel.com)
- Import your GitHub repository
- Configure build settings (see vercel.json)
- Deploy with one click

## 📁 Project Structure

```
mental-health-chatbot/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── vercel.json           # Vercel deployment configuration
├── runtime.txt           # Python version specification
├── .gitignore           # Git ignore rules
├── README.
