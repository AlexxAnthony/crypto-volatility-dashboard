#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

@st.cache_data
def load_real_data():
    return pd.read_csv("volatility_forecasting.csv")

df = load_real_data()

st.set_page_config(page_title="Technical Deep Dive", layout="wide")
st.title("🔬 Technical Deep Dive - Real Data")

# FEATURE ANALYSIS
st.header("📊 Feature Analysis")
if 'sentiment_score' in df.columns:
    # Correlation heatmap
    numeric_df = df[['actual_volatility', 'predicted_volatility', 'sentiment_score']].corr()
    fig = px.imshow(numeric_df, text_auto=True, aspect="auto",
                   title="Correlation Between Key Variables")
    st.plotly_chart(fig)

# ERROR ANALYSIS
st.header("📈 Error Distribution")
errors = df['actual_volatility'] - df['predicted_volatility']
fig = px.histogram(x=errors, nbins=50, 
                   title="Distribution of Prediction Errors",
                   labels={'x': 'Prediction Error', 'y': 'Frequency'})
st.plotly_chart(fig)

# MODEL PERFORMANCE OVER TIME
st.header("⏰ Performance Over Time")
if 'timestamp' in df.columns:
    df['error'] = errors.abs()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    daily_error = df.groupby(df['timestamp'].dt.date)['error'].mean()
    
    fig = px.line(x=daily_error.index, y=daily_error.values,
                 title="Daily Average Prediction Error",
                 labels={'x': 'Date', 'y': 'Mean Absolute Error'})
    st.plotly_chart(fig)

# QUANTILE ANALYSIS
st.header("📶 Performance by Volatility Level")
df['volatility_quantile'] = pd.qcut(df['actual_volatility'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
quantile_performance = df.groupby('volatility_quantile')['error'].mean()

fig = px.bar(x=quantile_performance.index, y=quantile_performance.values,
             title="Prediction Error by Volatility Level",
             labels={'x': 'Volatility Level', 'y': 'Mean Error'})
st.plotly_chart(fig)


# In[3]:


# Add to imports
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Add after existing sections
st.header("🧠 Model Architecture")
st.write("XGBoost Regressor with sentiment and volatility features")

# Display model parameters
st.subheader("Model Parameters")
st.code("""
XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)
""")

# Add feature importance visualization
st.header("📊 Feature Importance")
feature_importance = pd.DataFrame({
    'feature': ['realized_volatility_24h', 'sentiment_score'],
    'importance': [0.607696, 0.392304]
})
fig = px.bar(feature_importance, x='feature', y='importance', 
             title='XGBoost Feature Importance')
st.plotly_chart(fig)


# In[18]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

# -----------------------------------------
# Page Config
# -----------------------------------------
st.set_page_config(page_title="📈 Technical Deep Dive", layout="wide")
st.title("📈 Technical Deep Dive")
st.markdown("For technical reviewers, data scientists, and quants.")

# -----------------------------------------
# Load Files with better error handling
# -----------------------------------------
@st.cache_data
def load_all():
    try:
        model = joblib.load("xgboost_model.joblib")
        scaler = joblib.load("scaler.joblib")
        X = pd.read_csv("X_data.csv")
        
        # Debug info
        st.sidebar.info(f"Model type: {type(model)}")
        st.sidebar.info(f"X shape: {X.shape}")
        st.sidebar.info(f"X columns: {list(X.columns)}")
        
        return model, scaler, X
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

model, scaler, X = load_all()

if model is None or X is None:
    st.stop()

# -----------------------------------------
# Debug: Check model and data compatibility
# -----------------------------------------
st.sidebar.header("🔍 Debug Info")

# Check if model has been fitted
if hasattr(model, 'n_features_in_'):
    st.sidebar.info(f"Model expects {model.n_features_in_} features")
    st.sidebar.info(f"Data has {X.shape[1]} features")
    
    if model.n_features_in_ != X.shape[1]:
        st.sidebar.error("Feature count mismatch!")
        st.sidebar.info(f"Model: {model.n_features_in_}, Data: {X.shape[1]}")

st.header("💻 4. Code Snippets")

with st.expander("📌 Model Training Code"):
    st.code("""
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)
model.fit(X_train, y_train)
    """, language="python")

with st.expander("📌 SHAP Plot Code"):
    st.code("""
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample_scaled)
shap.summary_plot(shap_values, X_sample_scaled, feature_names=X.columns)
    """, language="python")


# In[ ]:




