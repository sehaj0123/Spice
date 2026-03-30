import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="SPICE Solar Dashboard", layout="wide")

st.title("☀️ SPICE Solar Energy Dashboard")

page = st.sidebar.selectbox(
    "Select Section",
    ["Overview", "EDA", "Modeling", "XAI"]
)

df = pd.read_csv("final_cleaned_dataset.csv")

if page == "Overview":
    st.header("Project Overview")
    st.write("This project includes EDA, modeling, and XAI.")

elif page == "EDA":
    st.header("EDA")

    monthly_total = df.groupby("month")["combined_production"].mean().sort_index()

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    labels = [month_names[m-1] for m in monthly_total.index]

    fig, ax = plt.subplots()
    ax.bar(labels, monthly_total.values)
    ax.set_title("Monthly Solar Production (Full Year)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Production")

    st.pyplot(fig)

elif page == "Modeling":
    st.header("Model")
    st.write("Random Forest Model (R² ≈ 0.67)")

elif page == "XAI":
    st.header("XAI")
    st.write("Feature importance, actual vs predicted, and residual plots were created in the notebook.")
