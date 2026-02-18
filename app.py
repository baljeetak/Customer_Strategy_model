import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration
st.set_page_config(page_title="Customer Strategy AI", page_icon="🎯", layout="wide")

# 2. Advanced CSS to fix Padding and Force Black Fonts
st.markdown("""
    <style>
    .block-container { padding-top: 3.5rem !important; padding-bottom: 0rem !important; }
    .main { background-color: #f8f9fa; }
    
    /* Force ALL text to be Black */
    .prediction-card, .strategy-card, [data-testid="stMetricLabel"], [data-testid="stMetricValue"], .stMarkdown {
        color: black !important;
    }
    
    .prediction-card {
        padding: 1.2rem; border-radius: 0.5rem; background-color: white;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075); border-top: 5px solid #4e73df;
        margin-bottom: 0.8rem;
    }

    .strategy-card {
        padding: 1.2rem; border-radius: 0.5rem; background-color: #ffffff;
        border-left: 5px solid #333; box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        margin-top: 0.8rem;
    }

    .stMetric { background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Assets
@st.cache_resource
def load_assets():
    try:
        with open('kmeans_model.pkl', 'rb') as f: model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
        df = pd.read_csv('Mall_Customers_dataset.csv')
        return model, scaler, df
    except: return None, None, None

model, scaler, df = load_assets()

# 4. Sidebar Inputs
with st.sidebar:
    st.header("📍 Customer Profile")
    income = st.slider("Annual Income (k$)", 10, 150, 60)
    score = st.slider("Spending Score (1-100)", 1, 100, 50)
    st.divider()
    # st.caption("AI Model created by Akash.")

# 5. Prediction Logic
user_input = np.array([[income, score]])
user_scaled = scaler.transform(user_input)
cluster_id = model.predict(user_scaled)[0]

# --- AUTO-DETECTION LOGIC (Fixes the "High/Low" swap forever) ---
# We find which cluster ID belongs to which behavior by looking at the centers
centers = scaler.inverse_transform(model.cluster_centers_)
profile_map = {}

for i, center in enumerate(centers):
    c_inc, c_score = center[0], center[1]
    if c_inc > 70 and c_score > 70:
        profile_map[i] = {"name": "Target (Elite)", "color": "#1f77b4", "desc": "High Income & High Spending", "strat": "💎 VIP Treatment: Exclusive previews and rewards."}
    elif c_inc < 45 and c_score < 45:
        profile_map[i] = {"name": "Sensible (Budget)", "color": "#9467bd", "desc": "Low Income & Low Spending", "strat": "🏷️ Value Focus: Buy 1 Get 1 and coupons."}
    elif c_inc < 45 and c_score > 70:
        profile_map[i] = {"name": "Spendthrifts", "color": "#d62728", "desc": "Low Income & High Spending", "strat": "⚡ Flash Sales: Use social media trends and impulse offers."}
    elif c_inc > 70 and c_score < 45:
        profile_map[i] = {"name": "Careful (Savers)", "color": "#2ca02c", "desc": "High Income & Low Spending", "strat": "💰 Premium Incentives: High-value vouchers to trigger purchases."}
    else:
        profile_map[i] = {"name": "Standard (Average)", "color": "#ff7f0e", "desc": "Average Income & Spending", "strat": "📩 Engagement: Regular updates and loyalty points."}

res = profile_map.get(cluster_id)

# 6. Main Dashboard Layout
st.title("📊 Customer Strategy Dashboard")
col1, col2 = st.columns([1, 1.8], gap="medium")

with col1:
    st.markdown(f"""
        <div class="prediction-card">
            <p style="margin:0; font-size: 0.8rem; color: #666; text-transform: uppercase;">Result</p>
            <h2 style='color: {res['color']} !important;'>{res['name']}</h2>
            <p style="margin-top:5px; color: black;"><b>Profile:</b> {res['desc']}</p>
            <p style="color: black;">Classified: <b>Cluster ID {cluster_id}</b></p>
        </div>
    """, unsafe_allow_html=True)
    
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Income", f"${income}k")
    m_col2.metric("Score", f"{score}/100")

    st.markdown(f"""
        <div class="strategy-card">
            <h4 style="color: black;">🚀 Marketing Action:</h4>
            <p style="color: black;">{res['strat']}</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.patch.set_facecolor('#f8f9fa')
    
    X_raw = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    y_kmeans = model.predict(scaler.transform(X_raw))
    
    for i in range(5):
        ax.scatter(X_raw[y_kmeans == i, 0], X_raw[y_kmeans == i, 1], 
                   s=60, c=profile_map[i]['color'], label=profile_map[i]['name'], alpha=0.15, edgecolors='none')
    
    ax.scatter(centers[:, 0], centers[:, 1], s=120, c='black', marker='X', label='Centers')
    ax.scatter(income, score, s=450, c='yellow', marker='*', edgecolors='black', linewidth=2, label='Current')
    
    ax.set_title('Live Segmentation Map', fontsize=14, color='black', pad=10)
    ax.set_xlabel('Annual Income', color='black')
    ax.set_ylabel('Spending Score', color='black')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    st.pyplot(fig)