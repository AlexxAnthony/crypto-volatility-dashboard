#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import time

# Load YOUR REAL data for reference
@st.cache_data
def load_real_data():
    return pd.read_csv("volatility_forecasting.csv")

df = load_real_data()

st.set_page_config(page_title="Live Market", layout="wide")
st.title("⚡ Live Market Simulation")

# LIVE


# In[2]:


# Add to imports
import requests
import time
from datetime import datetime

# Replace the empty LIVE section with:
st.header("📈 Real-time Crypto Prices")

# Function to fetch live prices
@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_live_prices():
    try:
        # Use a free crypto API like CoinGecko
        response = requests.get("https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=10&page=1&sparkline=false")
        data = response.json()
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

# Function to get news sentiment (simplified)
def analyze_news_sentiment():
    # This would connect to a news API in a real implementation
    return np.random.uniform(-1, 1)  # Placeholder

# Create layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Live Prices")
    prices_df = get_live_prices()
    if not prices_df.empty:
        st.dataframe(prices_df[['name', 'current_price', 'price_change_percentage_24h']])
    else:
        st.warning("Could not fetch live data")

with col2:
    st.subheader("Market Sentiment")
    sentiment = analyze_news_sentiment()
    st.metric("Current Sentiment", f"{sentiment:.2f}", 
              "Positive" if sentiment > 0 else "Negative")
    
    # Sentiment history chart
    sentiment_history = st.session_state.get('sentiment_history', [])
    sentiment_history.append(sentiment)
    if len(sentiment_history) > 50:
        sentiment_history.pop(0)
    st.session_state.sentiment_history = sentiment_history
    
    fig = px.line(y=sentiment_history, title="Sentiment Over Time")
    st.plotly_chart(fig)

# Add auto-refresh
if st.button("Refresh Data"):
    st.rerun()

# Auto-refresh every 60 seconds
st.write("Data refreshes every 60 seconds")
time.sleep(60)
st.rerun()


# In[ ]:




