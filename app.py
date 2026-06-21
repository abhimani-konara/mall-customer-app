import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from PIL import Image

st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛒",
    layout="wide"
)

# Train model directly in app
@st.cache_resource
def load_model():
    df = pd.read_csv("Dataset_3.csv")
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    return df, scaler, kmeans

df, scaler, model = load_model()

cluster_info = {
    0: {"name": "Average Customers", "color": "🔴", "profile": "Middle-aged, average income, average spending", "strategy": "General membership with loyalty points and monthly discounts"},
    1: {"name": "High Value Customers ⭐", "color": "🔵", "profile": "Young, high income, high spenders", "strategy": "Premium Gold Membership — VIP perks, exclusive events, priority access"},
    2: {"name": "Young Spenders", "color": "🟢", "profile": "Young, low income, high spenders", "strategy": "Student/Budget Membership — installment plans, buy-now-pay-later deals"},
    3: {"name": "Careful Spenders", "color": "🟠", "profile": "Middle-aged, high income, low spenders", "strategy": "Savings-focused Membership — cashback rewards, exclusive deals"},
    4: {"name": "Low Activity Customers", "color": "🟣", "profile": "Older, low income, low spenders", "strategy": "Basic Discount Membership — senior discounts, essential item deals"}
}

st.sidebar.title("🛒 Mall Customer App")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "📊 Dataset Overview",
    "📈 EDA & Visualizations",
    "🤖 Cluster Results",
    "🎯 Predict My Cluster"
])

if page == "🏠 Home":
    st.title("🛒 Mall Customer Segmentation")
    st.subheader("Machine Learning Project — Group 08 | HNDCSAI26.1")
    st.markdown("**NIBM City University (Kandy Innovation Center) | HND School of Computing**")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", "200")
    col2.metric("Features Used", "3")
    col3.metric("Clusters Found", "5")
    st.markdown("---")
    st.markdown("### 🎯 Problem Statement")
    st.info("How can different clustering techniques help a supermarket increase their membership card conversion rate by identifying customer segments with similar shopping preferences and purchasing behaviour?")
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
    t1.markdown("**KIC-HNDCSAI-26.1-F-05**\n\n_(Name here)_")
    t2.markdown("**KIC-HNDCSAI-26.1-F-07**\n\n_(Name here)_")
    t3.markdown("**KIC-HNDCSAI-26.1-F-028**\n\n_(Name here)_")

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
    st.markdown("### 📈 Basic Statistics")
    st.dataframe(df.describe(), use_container_width=True)

elif page == "📈 EDA & Visualizations":
    st.title("📈 EDA & Visualizations")
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["Heatmap", "Pairplot", "Cluster Charts", "Gender Analysis"])
    with tab1:
        st.subheader("Feature Correlation Heatmap")
        try:
            st.image(Image.open("heatmap.png"), use_column_width=True)
        except:
            st.warning("heatmap.png not found.")
        st.markdown("**Insight:** Spending Score and Age have a moderate negative correlation (-0.33)")
    with tab2:
        st.subheader("Pairplot")
        try:
            st.image(Image.open("pairplot.png"), use_column_width=True)
        except:
            st.warning("pairplot.png not found.")
    with tab3:
        st.subheader("Cluster Charts")
        c1, c2 = st.columns(2)
        with c1:
            try:
                st.image(Image.open("cluster_pie.png"), use_column_width=True)
            except:
                st.warning("cluster_pie.png not found.")
        with c2:
            try:
                st.image(Image.open("cluster_bar_comparison.png"), use_column_width=True)
            except:
                st.warning("cluster_bar_comparison.png not found.")
    with tab4:
        st.subheader("Gender Analysis")
        try:
            st.image(Image.open("gender_cluster.png"), use_column_width=True)
        except:
            st.warning("gender_cluster.png not found.")

elif page == "🤖 Cluster Results":
    st.title("🤖 Cluster Results")
    st.markdown("---")
    try:
        st.image(Image.open("model_comparison.png"), use_column_width=True)
    except:
        pass
    comp_df = pd.DataFrame({
        "Model": ["K-Means", "Hierarchical", "DBSCAN"],
        "Clusters": [5, 5, "2 + noise"],
        "Silhouette Score": [0.5547, 0.5538, 0.3504],
        "Verdict": ["✅ Best", "✅ Good", "⚠️ Lower"]
    })
    st.table(comp_df)
    st.markdown("### 5 Customer Segments")
    for cluster_id, info in cluster_info.items():
        with st.expander(f"{info['color']} Cluster {cluster_id} — {info['name']}"):
            col1, col2 = st.columns(2)
            col1.markdown(f"**Profile:** {info['profile']}")
            col2.markdown(f"**Strategy:** {info['strategy']}")

elif page == "🎯 Predict My Cluster":
    st.title("🎯 Predict Customer Cluster")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", min_value=18, max_value=70, value=30)
        annual_income = st.slider("Annual Income (k$)", min_value=15, max_value=140, value=60)
        spending_score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)
    with col2:
        st.markdown("### Customer Summary")
        st.info(f"**Age:** {age} years\n\n**Annual Income:** ${annual_income}k\n\n**Spending Score:** {spending_score}/100")
    st.markdown("---")
    if st.button("🔍 Predict Cluster", use_container_width=True):
        input_data = np.array([[age, annual_income, spending_score]])
        input_scaled = scaler.transform(input_data)
        cluster = model.predict(input_scaled)[0]
        info = cluster_info[cluster]
        st.success(f"## {info['color']} Cluster {cluster} — {info['name']}")
        st.markdown(f"**Profile:** {info['profile']}")
        st.markdown(f"**Strategy:** {info['strategy']}")
        st.balloons()