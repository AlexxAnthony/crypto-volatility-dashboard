#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st

# Set page configuration (MUST be the first Streamlit command)
st.set_page_config(
    page_title="Crypto Volatility Forecasting App",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main content
st.title("🚀 Welcome to the Crypto Volatility Forecasting App")
st.markdown("""
This app includes multiple interactive dashboards:

- 🌟 **Executive Summary**: High-level results and impact
- 🔍 **Sentiment Analyzer**: Try live headline analysis  
- 📈 **Technical Deep Dive**: Explore modeling and SHAP
- ⚡ **Live Market Dashboard**: Real-time crypto data

**Use the sidebar to navigate ➡️**
""")

# Add some visual elements
col1, col2 = st.columns(2)
with col1:
    st.info("**Real-time Analysis**")
    st.write("Get live market insights and sentiment data")
    
with col2:
    st.info("**Advanced Forecasting**")
    st.write("Machine learning models for volatility prediction")

# Optional: Add some instructions
st.success("💡 **Tip**: Select a page from the sidebar on the left to explore different features!")


# In[ ]:




