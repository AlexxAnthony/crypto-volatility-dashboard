#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import plotly.express as px

# Load YOUR REAL data
@st.cache_data
def load_real_data():
    return pd.read_csv("volatility_forecasting.csv")

df = load_real_data()

st.set_page_config(page_title="Sentiment Analyzer", layout="wide")
st.title("🔮 Real Sentiment Analysis")

# SENTIMENT DISTRIBUTION
st.header("📊 Real Sentiment Distribution")
if 'sentiment_score' in df.columns:
    fig = px.histogram(df, x='sentiment_score', 
                       title='Distribution of Your Real Sentiment Scores',
                       nbins=30)
    st.plotly_chart(fig)

# SENTIMENT vs VOLATILITY
st.header("📈 Sentiment vs Volatility Relationship")
if 'sentiment_score' in df.columns:
    fig = px.scatter(df.sample(n=1000), x='sentiment_score', y='actual_volatility',
                     title='How Sentiment Affects Actual Volatility',
                     trendline='ols')
    st.plotly_chart(fig)

# INTERACTIVE SENTIMENT ANALYSIS
st.header("🎯 Try Live Sentiment Analysis")
news_headline = st.text_input("Enter cryptocurrency news headline:",
                             "Elon Musk announces Bitcoin integration with Tesla")

positive_words = ['buy', 'bullish', 'growth', 'adopt', 'integrate', 'announce', 'partner']
negative_words = ['sell', 'bearish', 'crash', 'drop', 'warning', 'risk', 'scam']

if news_headline:
    positive_count = sum(1 for word in positive_words if word in news_headline.lower())
    negative_count = sum(1 for word in negative_words if word in news_headline.lower())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Positive Words", positive_count)
    with col2:
        st.metric("Negative Words", negative_count)
    
    if positive_count > negative_count:
        st.success("📈 Predicted: POSITIVE Market Impact")
    elif negative_count > positive_count:
        st.error("📉 Predicted: NEGATIVE Market Impact")
    else:
        st.info("🤔 Predicted: NEUTRAL Market Impact")


# In[ ]:




