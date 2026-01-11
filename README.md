🌆 Urban Stress Index Analytics Dashboard
📌 Project Overview

Urbanization significantly impacts human well-being through factors such as pollution, climate conditions, infrastructure load, population density, and green space availability.
This project presents an Urban Stress Index Dashboard that analyzes and visualizes stress levels of cities using unsupervised machine learning (K-Means clustering) and interactive data visualization.

The application enables:

City-level stress classification

Data-driven urban stress scoring

Geo-spatial visualization

Scenario-based What-If stress prediction

Built and deployed using Streamlit.

🎯 Objectives

Analyze multi-dimensional urban factors contributing to city stress

Create a normalized Urban Stress Index (0–1 scale)

Categorize cities into meaningful stress levels

Provide interactive analytics for policy makers and planners

Enable real-time stress prediction using hypothetical inputs

🧠 Key Features

✔️ Exploratory Data Analysis (EDA)
✔️ Feature Scaling (StandardScaler, MinMaxScaler)
✔️ K-Means Clustering (5 stress clusters)
✔️ Stress Level Classification (Very Low → Extreme)
✔️ Urban Stress Index computation
✔️ Interactive Streamlit Dashboard
✔️ Geo-spatial City Stress Map with Legend
✔️ Cluster-wise Comparison Charts
✔️ Button-based What-If Stress Prediction
✔️ Clean UI/UX with advanced visualizations

🗂️ Dataset Information

The project uses a merged dataset containing:

Air Pollution: PM2.5, PM10, NO₂, O₃, SO₂, CO

Climate: Temperature, Humidity

Infrastructure: Roads, Transit Stations, Parking

Demographics: Population

Environment: Green Spaces

📁 File: urban_stress_index.csv

🧪 Methodology
1️⃣ Data Preprocessing

Missing value handling

Feature selection

Standardization using StandardScaler

2️⃣ Clustering

K-Means (k=5 chosen via Elbow Method)

Clusters represent stress groupings

3️⃣ Urban Stress Index Creation

A composite index based on:

Pollution + Temperature + Population – Green Spaces


Normalized to 0–1 scale using MinMaxScaler.

4️⃣ Stress Level Classification

Clusters are ranked by stress index and mapped to:

Very Low Stress

Low Stress

Moderate Stress

High Stress

Extreme Stress

🧪 What-If Stress Prediction

Users can simulate urban scenarios by changing:

Population

Pollution levels

Temperature

Infrastructure

Green spaces

Predictions are generated only after clicking a button, ensuring controlled evaluation.

🗺️ Visualizations Included

Stress Level Distribution

Pollution Analysis per City

Cluster-wise Comparisons

Correlation Heatmap

Green Spaces vs Stress Scatter Plot

Geo-Spatial Stress Map (Folium)

Top 10 Most Stressed Cities

🛠️ Tech Stack

Python

Pandas / NumPy

Scikit-Learn

Matplotlib / Seaborn

Streamlit

Folium & Streamlit-Folium

GitHub

Streamlit Community Cloud

🚀 How to Run Locally
1️⃣ Clone Repository
git clone https://github.com/your-username/urban-stress-index-dashboard.git
cd urban-stress-index-dashboard

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Application
streamlit run app.py

🌐 Live Deployment

🚀 Deployed on Streamlit Community Cloud

🔗 Live App URL:
(Add your deployed link here)

📈 Use Cases

Urban Planning & Policy Analysis

Environmental Risk Assessment

Smart City Analytics

Academic & Research Projects

Data Analyst Portfolio Demonstration

🔮 Future Enhancements

Time-series stress trend analysis

Integration of real-time pollution APIs

Supervised ML stress prediction model

City-to-City comparison dashboards

Policy recommendation engine

👤 Author

Abi Shek
Data Analyst | Machine Learning Enthusiast | Full-Stack Analytics Developer
