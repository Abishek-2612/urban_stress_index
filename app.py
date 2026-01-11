# ==================================================
# 🌆 URBAN STRESS INDEX DASHBOARD (5/5 VERSION)
# ==================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium

# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Urban Stress Index Dashboard",
    layout="wide"
)

st.title("🌆 Urban Stress Index Dashboard")
st.markdown("""
This dashboard analyzes **urban stress levels** using pollution, climate,
infrastructure, population, and green space indicators.
""")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("urban_stress_index.csv")

df = load_data()

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# FEATURE SELECTION
# --------------------------------------------------
features = [
    'population','avg_pm25','avg_pm10','avg_no2',
    'temperature','humidity','total_infrastructure','num_green_spaces'
]

X = df[features]

# --------------------------------------------------
# SCALING
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# KMEANS CLUSTERING
# --------------------------------------------------
kmeans = KMeans(n_clusters=5, random_state=42)
df['stress_cluster'] = kmeans.fit_predict(X_scaled)

# --------------------------------------------------
# DATA-DRIVEN URBAN STRESS INDEX
# --------------------------------------------------
stress_features = [
    'avg_pm25','avg_pm10','avg_no2','temperature','population','num_green_spaces'
]

cluster_summary = df.groupby('stress_cluster')[stress_features].mean()

# Raw stress score
cluster_summary['raw_stress_score'] = (
    cluster_summary['avg_pm25'] +
    cluster_summary['avg_pm10'] +
    cluster_summary['avg_no2'] +
    cluster_summary['temperature'] +
    (cluster_summary['population'] / 1_000_000) -
    cluster_summary['num_green_spaces']
)

# Normalize stress score
index_scaler = MinMaxScaler()
cluster_summary['urban_stress_index'] = index_scaler.fit_transform(
    cluster_summary[['raw_stress_score']]
)

# Map index back to cities
stress_index_map = cluster_summary['urban_stress_index'].to_dict()
df['urban_stress_index'] = df['stress_cluster'].map(stress_index_map)

# Assign stress level labels
sorted_clusters = cluster_summary.sort_values('urban_stress_index').index.tolist()
stress_labels = ["Very Low Stress", "Low Stress", "Moderate Stress", "High Stress", "Extreme Stress"]
cluster_label_map = dict(zip(sorted_clusters, stress_labels))
df['stress_level_ml'] = df['stress_cluster'].map(cluster_label_map)

# ----------------------------
# City Filters
# ----------------------------
st.sidebar.header("📌 Filters")
region_options = ["All"] + sorted(df['region'].unique().tolist())
selected_region = st.sidebar.selectbox("Filter by Region", region_options)

stress_filter = st.sidebar.multiselect("Filter by Stress Level", stress_labels, default=stress_labels)

filtered_df = df.copy()
if selected_region != "All":
    filtered_df = filtered_df[filtered_df['region'] == selected_region]
filtered_df = filtered_df[filtered_df['stress_level_ml'].isin(stress_filter)]

st.subheader("📊 Filtered Cities Overview")
st.dataframe(filtered_df[['city','region','stress_level_ml','urban_stress_index']], use_container_width=True)

# --------------------------------------------------
# STRESS LEVEL DISTRIBUTION
# --------------------------------------------------
st.subheader("📊 City Stress Level Distribution")
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(
    data=filtered_df,
    x='stress_level_ml',
    order=stress_labels,
    palette=['green','lime','yellow','orange','red'],
    ax=ax
)
ax.set_xlabel("Stress Level")
ax.set_ylabel("Number of Cities")
st.pyplot(fig)
st.caption("📌 Insight: Higher stress levels are associated with higher pollution and lower green spaces.")

# --------------------------------------------------
# SIDEBAR CITY SELECTION
# --------------------------------------------------
st.sidebar.header("🔍 City Selection")
city_list = sorted(filtered_df['city'].unique())
selected_city = st.sidebar.selectbox("Select a City", city_list)
city_data = filtered_df[filtered_df['city'] == selected_city].iloc[0]

# --------------------------------------------------
# CITY OVERVIEW
# --------------------------------------------------
st.subheader(f"📍 City Overview: {selected_city}")
col1, col2, col3 = st.columns(3)
col1.metric("Stress Level", city_data['stress_level_ml'])
col2.metric("Urban Stress Index", round(city_data['urban_stress_index'], 2))
col3.metric("Population", int(city_data['population']))

# --------------------------------------------------
# POLLUTION ANALYSIS
# --------------------------------------------------
st.subheader("🏭 Pollution Levels")
pollution_cols = ['avg_pm25','avg_pm10','avg_no2','avg_o3','avg_so2','avg_co']
pollution_values = city_data[pollution_cols]

fig, ax = plt.subplots(figsize=(8, 4))
pollution_values.plot(kind='bar', ax=ax, color='crimson')
ax.set_ylabel("Concentration Level")
ax.set_xlabel("Pollutants")
st.pyplot(fig)

# --------------------------------------------------
# ENVIRONMENT & INFRASTRUCTURE
# --------------------------------------------------
st.subheader("🌳 Environment & Infrastructure")
col4, col5, col6 = st.columns(3)
col4.metric("Green Spaces", city_data['num_green_spaces'])
col5.metric("Infrastructure Index", city_data['total_infrastructure'])
col6.metric("Temperature (°C)", city_data['temperature'])

# --------------------------------------------------
# OVERALL INSIGHTS
# --------------------------------------------------
st.header("📊 Overall Urban Stress Insights")
top_stress = filtered_df.sort_values(by='urban_stress_index', ascending=False).head(10)
st.subheader("🚨 Top 10 Most Stressed Cities")
st.dataframe(top_stress[['city','region','urban_stress_index','stress_level_ml']], use_container_width=True)

# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------
st.subheader("🔗 Correlation Between Urban Factors")
corr_features = ['population','avg_pm25','temperature','total_infrastructure','num_green_spaces','urban_stress_index']
corr = filtered_df[corr_features].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot(fig)
st.caption("📌 Insight: Pollution and population are strongly correlated with urban stress index.")

# --------------------------------------------------
# GREEN SPACES VS URBAN STRESS
# --------------------------------------------------
st.subheader("🌿 Green Spaces vs Urban Stress")
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    data=filtered_df,
    x='num_green_spaces',
    y='urban_stress_index',
    hue='stress_level_ml',
    palette=['green','lime','yellow','orange','red'],
    ax=ax
)
ax.set_xlabel("Number of Green Spaces")
ax.set_ylabel("Urban Stress Index")
st.pyplot(fig)
st.caption("📌 Insight: Cities with more green spaces generally have lower urban stress.")

# --------------------------------------------------
# GEO VISUALIZATION
# --------------------------------------------------
st.header("🗺️ Urban Stress Map")
def stress_color(level):
    return {
        "Very Low Stress": "green",
        "Low Stress": "lime",
        "Moderate Stress": "orange",
        "High Stress": "red",
        "Extreme Stress": "darkred"
    }[level]

m = folium.Map(location=[20, 78], zoom_start=4)
legend_html = """
<div style="position: fixed; bottom: 50px; left: 50px; width: 220px;
background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
padding: 10px; border-radius: 8px;">
<b>Urban Stress Level</b><br>
<span style="color:green;">●</span> Very Low Stress<br>
<span style="color:lime;">●</span> Low Stress<br>
<span style="color:orange;">●</span> Moderate Stress<br>
<span style="color:red;">●</span> High Stress<br>
<span style="color:darkred;">●</span> Extreme Stress
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

for _, row in filtered_df.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=6,
        color=stress_color(row['stress_level_ml']),
        fill=True,
        fill_color=stress_color(row['stress_level_ml']),
        fill_opacity=0.7,
        popup=f"<b>City:</b> {row['city']}<br><b>Stress Level:</b> {row['stress_level_ml']}<br><b>Urban Stress Index:</b> {round(row['urban_stress_index'],2)}"
    ).add_to(m)

st_folium(m, width=1200, height=600)

# --------------------------------------------------
# CLUSTER COMPARISON CHARTS
# --------------------------------------------------
compare_features = ['avg_pm25','avg_pm10','temperature','num_green_spaces','total_infrastructure','urban_stress_index']
cluster_summary = df.groupby('stress_level_ml')[compare_features].mean().reset_index()

st.subheader("🏭 Pollution Levels Across Stress Clusters")
fig, ax = plt.subplots(figsize=(10, 5))
cluster_summary.set_index('stress_level_ml')[['avg_pm25','avg_pm10']].plot(kind='bar', ax=ax)
plt.xticks(rotation=15)
ax.set_ylabel("Pollution Concentration")
ax.set_xlabel("Stress Level")
st.pyplot(fig)

st.subheader("🌿 Green Spaces vs Infrastructure Across Stress Clusters")
fig, ax = plt.subplots(figsize=(10, 5))
cluster_summary.set_index('stress_level_ml')[['num_green_spaces','total_infrastructure']].plot(kind='bar', ax=ax)
plt.xticks(rotation=15)
ax.set_ylabel("Count / Index")
ax.set_xlabel("Stress Level")
st.pyplot(fig)

st.subheader("📈 Stress Factor Trend Across Clusters")
fig, ax = plt.subplots(figsize=(10, 5))
cluster_summary.plot(x='stress_level_ml', y='urban_stress_index', marker='o', ax=ax)
ax.set_xlabel("Stress Level")
ax.set_ylabel("Urban Stress Index")
st.pyplot(fig)

# --------------------------------------------------
# WHAT-IF STRESS PREDICTION (BUTTON BASED)
# --------------------------------------------------
st.sidebar.header("🔮 What-If Stress Prediction")

user_inputs = [
    st.sidebar.number_input("Population", 1_000, 10_000_000, 500_000),
    st.sidebar.number_input("Avg PM2.5", 0.0, 500.0, 50.0),
    st.sidebar.number_input("Avg PM10", 0.0, 500.0, 40.0),
    st.sidebar.number_input("Avg NO2", 0.0, 200.0, 40.0),
    st.sidebar.number_input("Temperature (°C)", -10.0, 50.0, 25.0),
    st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 50.0),
    st.sidebar.number_input("Infrastructure Index", 0, 100, 50),
    st.sidebar.number_input("Green Spaces", 0, 500, 10)
]

predict_btn = st.sidebar.button("🚀 Predict Stress Level")

if predict_btn:
    user_scaled = scaler.transform([user_inputs])
    pred_cluster = kmeans.predict(user_scaled)[0]
    pred_label = cluster_label_map[pred_cluster]
    pred_index = stress_index_map[pred_cluster]
    st.subheader("🧪 What-If Prediction Result")
    st.success(f"Predicted Stress Level: **{pred_label}**")
    st.info(f"Urban Stress Index: **{round(pred_index,2)}**")

    color_map = {
        "Very Low Stress":"green",
        "Low Stress":"lime",
        "Moderate Stress":"yellow",
        "High Stress":"orange",
        "Extreme Stress":"red"
    }
    st.markdown(f"<h3 style='color:{color_map[pred_label]}'>Predicted Stress Level: {pred_label}</h3>", unsafe_allow_html=True)

# --------------------------------------------------
# EXPORT DATA
# --------------------------------------------------
st.sidebar.download_button(
    label="⬇️ Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="urban_stress_filtered.csv",
    mime="text/csv"
)

