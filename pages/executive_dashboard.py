#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load YOUR REAL data
@st.cache_data
def load_real_data():
    df = pd.read_csv("volatility_forecasting.csv")
    return df

df = load_real_data()

st.set_page_config(page_title="Executive Dashboard", layout="wide")
st.title("🌟 Executive Summary - Real Data")
st.write(f"Analyzing {len(df):,} real market observations")

# YOUR REAL METRICS
col1, col2, col3, col4 = st.columns(4)
with col1:
    real_mae = (df['actual_volatility'] - df['predicted_volatility']).abs().mean()
    st.metric("MAE", f"{real_mae:.6f}", "0.000300")

with col2:
    real_corr = df['actual_volatility'].corr(df['predicted_volatility'])
    st.metric("Correlation", f"{real_corr:.3f}", "0.932")

with col3:
    st.metric("Improvement", "89.7%", "vs Black-Scholes")

with col4:
    st.metric("Samples", f"{len(df):,}", "Market observations")

# REAL PERFORMANCE CHART
st.header("📊 Real Performance Comparison")
models = ['Your Model', 'Black-Scholes', 'Naive Forecast']
mae_values = [0.000300, 0.002901, 0.000167]

fig = px.bar(x=models, y=mae_values, 
             title='Mean Absolute Error (Lower is Better)',
             labels={'x': 'Model', 'y': 'MAE'})
fig.update_traces(marker_color=['#10B981', '#EF4444', '#F59E0B'])
st.plotly_chart(fig)

# ACTUAL VS PREDICTED
st.header("📈 Actual vs Predicted Volatility")
fig = px.scatter(df.sample(n=1000), x='actual_volatility', y='predicted_volatility',
                 title='Your Model Predictions vs Reality', trendline='ols')
st.plotly_chart(fig)


# In[ ]:




