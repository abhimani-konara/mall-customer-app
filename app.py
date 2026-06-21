import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ── Page config ──────────────────────────────
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛒",
    layout="wide"
)

# ── Load model and scaler ────────────────────
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("Dataset_3.csv")

# ── Sidebar navigation ───────────────────────
st.sidebar.title("🛒 Mall Customer App")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "📊 Dataset Overview",
    "📈 EDA & Visualizations",
    "🤖 Cluster Results",
    "🎯 Predict My Cluster"
])

# ── Cluster descriptions ──────────────────────
cluster_info = {
    0: {
        "name": "Average Customers",
        "color": "🔴",
        "profile": "Middle-aged, average income, average spending",
        "strategy": "General membership with loyalty points and monthly discounts"
    },
    1: {
        "name": "High Value Customers ⭐",
        "color": "🔵",
        "profile": "Young, high income, high spenders",
        "strategy": "Premium Gold Membership — VIP perks, exclusive events, priority access"
    },
    2: {
        "name": "Young Spenders",
        "color": "🟢",
        "profile": "Young, low income, high spenders",
        "strategy": "Student/Budget Membership — installment plans, buy-now-pay-later deals"
    },
    3: {
        "name": "Careful Spenders",
        "color": "🟠",
        "profile": "Middle-aged, high income, low spenders",
        "strategy": "Savings-focused Membership — cashback rewards, exclusive deals"
    },
    4: {
        "name": "Low Activity Customers",
        "color": "🟣",
        "profile": "Older, low income, low spenders",
        "strategy": "Basic Discount Membership — senior discounts, essential item deals"
    }
}

# ════════════════════════════════════════════
# PAGE 1: HOME
# ════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🛒 Mall Customer Segmentation")
    st.subheader("Machine Learning Project — Group 05 | HNDCSAI26.1")
    st.markdown("**NIBM City University (Kandy Innovation Center) | HND School of Computing**")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", "200")
    col2.metric("Features Used", "3")
    col3.metric("Clusters Found", "5")

    st.markdown("---")
    st.markdown("### 🎯 Problem Statement")
    st.info(
        "How can different clustering techniques help a supermarket increase their "
        "membership card conversion rate by identifying customer segments with "
        "similar shopping preferences and purchasing behaviour?"
    )

    st.markdown("### 🤖 Models Used")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("**K-Means Clustering**\nSilhouette Score: 0.5547 ⭐ Best")
    with col2:
        st.info("**Hierarchical Clustering**\nSilhouette Score: 0.5538")
    with col3:
        st.warning("**DBSCAN Clustering**\nSilhouette Score: 0.3504")

    st.markdown("---")
    st.markdown("### 👥 Team Members — Group 08")
    t1, t2, t3 = st.columns(3)
    t1.markdown("**KIC-HNDCSAI-26.1-F-05**\n\n_(Chethani Thakshila )_")
    t2.markdown("**KIC-HNDCSAI-26.1-F-07**\n\n_(Isurika Bandara)_")
    t3.markdown("**KIC-HNDCSAI-26.1-F-028**\n\n_(Abhimani Konara)_")

# ════════════════════════════════════════════
# PAGE 2: DATASET OVERVIEW
# ════════════════════════════════════════════
elif page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Duplicates", df.duplicated().sum())

    st.markdown("### 👀 First 10 Rows")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### 📋 Feature Description")
    feature_desc = pd.DataFrame({
        "Feature": ["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"],
        "Type": ["Integer", "Categorical", "Integer", "Integer", "Integer"],
        "Description": [
            "Unique ID for each customer",
            "Male or Female",
            "Customer age (18–70)",
            "Annual income in thousands",
            "Score based on spending behaviour (1–100)"
        ]
    })
    st.table(feature_desc)

    st.markdown("### 📈 Basic Statistics")
    st.dataframe(df.describe(), use_container_width=True)

# ════════════════════════════════════════════
# PAGE 3: EDA & VISUALIZATIONS
# ════════════════════════════════════════════
elif page == "📈 EDA & Visualizations":
    st.title("📈 EDA & Visualizations")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Heatmap", "Pairplot", "Cluster Charts", "Gender Analysis"])

    with tab1:
        st.subheader("Feature Correlation Heatmap")
        try:
            img = Image.open("heatmap.png")
            st.image(img, use_column_width=True)
        except:
            st.warning("heatmap.png not found in images folder.")
        st.markdown("**Insight:** Spending Score and Age have a moderate negative correlation (-0.33), meaning younger customers tend to spend more.")

    with tab2:
        st.subheader("Pairplot — K-Means Clusters")
        try:
            img = Image.open("pairplot.png")
            st.image(img, use_column_width=True)
        except:
            st.warning("pairplot.png not found in images folder.")
        st.markdown("**Insight:** Clear cluster separation is visible when comparing Annual Income and Spending Score.")

    with tab3:
        st.subheader("Cluster Distribution Charts")
        c1, c2 = st.columns(2)
        with c1:
            try:
                st.image(Image.open("cluster_pie.png"), use_column_width=True)
                st.caption("Customer Distribution by Cluster")
            except:
                st.warning("cluster_pie.png not found.")
        with c2:
            try:
                st.image(Image.open("cluster_bar_comparison.png"), use_column_width=True)
                st.caption("Average Age, Income & Spending Score per Cluster")
            except:
                st.warning("cluster_bar_comparison.png not found.")

    with tab4:
        st.subheader("Gender Analysis Across Clusters")
        try:
            img = Image.open("gender_cluster.png")
            st.image(img, use_column_width=True)
        except:
            st.warning("gender_cluster.png not found.")
        st.markdown("**Insight:** Female customers dominate across all 5 clusters.")

# ════════════════════════════════════════════
# PAGE 4: CLUSTER RESULTS
# ════════════════════════════════════════════
elif page == "🤖 Cluster Results":
    st.title("🤖 Cluster Results")
    st.markdown("---")

    st.markdown("### Model Comparison")
    try:
        img = Image.open("model_comparison.png")
        st.image(img, use_column_width=True)
    except:
        pass

    comp_df = pd.DataFrame({
        "Model": ["K-Means", "Hierarchical", "DBSCAN"],
        "Clusters": [5, 5, "2 + noise"],
        "Silhouette Score": [0.5547, 0.5538, 0.3504],
        "Verdict": ["✅ Best — clear & scalable", "✅ Good — great for visualization", "⚠️ Lower — not ideal for this data"]
    })
    st.table(comp_df)

    st.markdown("---")
    st.markdown("### 5 Customer Segments Found")

    for cluster_id, info in cluster_info.items():
        with st.expander(f"{info['color']} Cluster {cluster_id} — {info['name']}"):
            col1, col2 = st.columns(2)
            col1.markdown(f"**Profile:** {info['profile']}")
            col2.markdown(f"**Marketing Strategy:** {info['strategy']}")

# ════════════════════════════════════════════
# PAGE 5: PREDICT MY CLUSTER
# ════════════════════════════════════════════
elif page == "🎯 Predict My Cluster":
    st.title("🎯 Predict Customer Cluster")
    st.markdown("Enter customer details below to find out which cluster they belong to.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", min_value=18, max_value=70, value=30)
        annual_income = st.slider("Annual Income (k$)", min_value=15, max_value=140, value=60)
        spending_score = st.slider("Spending Score (1–100)", min_value=1, max_value=100, value=50)

    with col2:
        st.markdown("### Customer Summary")
        st.info(f"""
        **Age:** {age} years
        **Annual Income:** ${annual_income}k
        **Spending Score:** {spending_score}/100
        """)

    st.markdown("---")

    if st.button("🔍 Predict Cluster", use_container_width=True):
        input_data = np.array([[age, annual_income, spending_score]])
        input_scaled = scaler.transform(input_data)
        cluster = model.predict(input_scaled)[0]

        info = cluster_info[cluster]
        st.success(f"## {info['color']} Cluster {cluster} — {info['name']}")
        st.markdown(f"**Customer Profile:** {info['profile']}")
        st.markdown(f"**Recommended Strategy:** {info['strategy']}")

        st.balloons()
