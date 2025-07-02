import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import datetime

# --- Page Config ---
st.set_page_config(page_title="Satellite-based PM Estimation", layout="wide")

# --- Title ---
st.title("🌍 Monitoring Air Pollution from Space")
st.markdown("**Using Satellite AOD, Ground Observations, Meteorological Data & AI/ML Models**")

# --- Sidebar ---
st.sidebar.header("📅 Select Date")
selected_date = st.sidebar.date_input("Date", datetime.date.today())

st.sidebar.header("📍 Select Region")
region = st.sidebar.selectbox("Region", ["India", "Delhi", "Mumbai", "Chennai", "Kolkata", "Bangalore"])

st.sidebar.header("📈 Options")
show_ground = st.sidebar.checkbox("Show Ground Station Comparison", value=True)
show_prediction = st.sidebar.checkbox("Show Predicted PM Map", value=True)

# --- Project Overview ---
st.subheader("🔍 Project Overview")
st.markdown("""
This dashboard visualizes **surface-level PM2.5 concentrations** estimated using:
- INSAT-derived **Aerosol Optical Depth (AOD)**
- **CPCB ground monitoring data**
- **MERRA-2 meteorological data**
- Trained using **Random Forest** and other ML techniques
""")

# --- PM2.5 Prediction Map ---
if show_prediction:
    st.subheader("🗺️ Predicted PM2.5 Concentration Map")

    if os.path.exists("sample_predicted_pm.csv"):
        pm_map_data = pd.read_csv("sample_predicted_pm.csv")

        fig = px.density_mapbox(
            pm_map_data, lat="lat", lon="lon", z="PM2.5",
            radius=10, center=dict(lat=22.0, lon=80.0), zoom=4,
            mapbox_style="carto-positron",
            title="Predicted PM2.5 Levels"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ 'sample_predicted_pm.csv' not found.")

# --- Ground Truth vs Prediction Plot ---
if show_ground:
    st.subheader("📊 Ground Truth vs Predicted PM2.5")

    if os.path.exists("sample_comparison.csv"):
        scatter_data = pd.read_csv("sample_comparison.csv")

        x = scatter_data["Actual_PM2.5"]
        y = scatter_data["Predicted_PM2.5"]
        m, b = np.polyfit(x, y, deg=1)
        regression_line = m * x + b

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=x, y=y, mode='markers',
            name='Data Points',
            marker=dict(color='blue', size=8)
        ))

        fig2.add_trace(go.Scatter(
            x=x, y=regression_line,
            mode='lines',
            name=f"y = {m:.2f}x + {b:.2f}",
            line=dict(color='red', width=2)
        ))

        fig2.update_layout(
            title="Model Validation: Actual vs Predicted PM2.5",
            xaxis_title="Actual PM2.5",
            yaxis_title="Predicted PM2.5",
            legend=dict(x=0.01, y=0.99),
            template="plotly_white"
        )

        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("⚠️ 'sample_comparison.csv' not found.")

# --- Download Section ---
st.subheader("📁 Download Predicted Data")
if os.path.exists("sample_predicted_pm.csv"):
    pm_map_data = pd.read_csv("sample_predicted_pm.csv")
    st.download_button(
        "Download Predicted PM2.5 CSV",
        data=pm_map_data.to_csv(index=False),
        file_name=f"predicted_pm_{selected_date}.csv",
        mime="text/csv"
    )
else:
    st.warning("Predicted data not available for download.")

# --- Footer ---
st.markdown("---")
st.markdown("🚀 Developed with ❤️ By team Codenauts | Powered by INSAT, CPCB, MERRA-2, and Machine Learning")
