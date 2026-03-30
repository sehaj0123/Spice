
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="SPICE Solar Dashboard", layout="wide")

st.title("☀️ SPICE Solar Energy Dashboard")

page = st.sidebar.selectbox(
    "Select Section",
    ["Overview", "EDA", "Modeling", "XAI"]
)

df = pd.read_csv("your_merged_dataset.csv")

if page == "Overview":
    st.header("Project Overview")
    st.write("This project includes EDA, modeling, and XAI.")

elif page == "EDA":
    st.header("EDA")

    fig, ax = plt.subplots()
    ax.scatter(df["solar_radiation"], df["Production"])
    st.pyplot(fig)

elif page == "Modeling":
    st.header("Model")
    st.write("Random Forest Model (R² ≈ 0.67)")

elif page == "XAI":
    st.header("XAI")

    features = ["solar_radiation", "Mean Temp", "Snow"]
    importance = [0.5, 0.3, 0.2]

    fig, ax = plt.subplots()
    ax.bar(features, importance)
    st.pyplot(fig)
