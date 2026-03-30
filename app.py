import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="SPICE Solar Dashboard", layout="wide")

st.title("☀️ SPICE Solar Energy Dashboard")

page = st.sidebar.selectbox(
    "Select Section",
    ["Overview", "EDA", "Business Analysis", "Modeling", "XAI"]
)

# Load files
business_df = pd.read_csv("final_cleaned_dataset.csv")
df = pd.read_csv("merged_without_price.csv")

# Fix dates
if "date" in business_df.columns:
    business_df["date"] = pd.to_datetime(business_df["date"])
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

# Make sure month exists
if "month" not in business_df.columns and "date" in business_df.columns:
    business_df["month"] = business_df["date"].dt.month

if "month" not in df.columns and "date" in df.columns:
    df["month"] = df["date"].dt.month

if "dayofyear" not in df.columns and "date" in df.columns:
    df["dayofyear"] = df["date"].dt.dayofyear

# Train model inside app for model + XAI visuals
features = [
    "solar_radiation",
    "solar_clear_sky",
    "solar_ratio",
    "wind_speed",
    "temperature_nasa_y",
    "Mean Temp (°C)",
    "Total Rain (mm)",
    "Total Snow (cm)",
    "month",
    "dayofyear"
]

available_features = [col for col in features if col in df.columns]

model_ready = all(col in df.columns for col in ["Production"] + available_features)

if model_ready:
    X = df[available_features]
    y = df["Production"]

    train_size = int(len(df) * 0.8)
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

if page == "Overview":
    st.header("Project Overview")
    st.write("""
    This project analyzes solar production using Visser, Bissell, weather,
    NASA solar radiation, and Alberta pool price data from AESO.

    The app includes:
    - Exploratory Data Analysis
    - Business Analysis
    - Machine Learning Modeling
    - Explainable AI (XAI)
    """)

elif page == "EDA":
    st.header("Exploratory Data Analysis")

    st.subheader("Monthly Total Solar Production")
    monthly_total = business_df.groupby("month")["combined_production"].mean().sort_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    labels = [month_names[m-1] for m in monthly_total.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, monthly_total.values)
    ax.set_title("Monthly Solar Production (Full Year)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Production")
    st.pyplot(fig)

    st.subheader("Solar Radiation vs Production")
    if "solar_radiation" in df.columns and "Production" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(df["solar_radiation"], df["Production"], alpha=0.6)
        ax.set_xlabel("Solar Radiation")
        ax.set_ylabel("Solar Production")
        ax.set_title("Solar Radiation vs Solar Production")
        st.pyplot(fig)

    st.subheader("Average Solar Production by Temperature Range")
    if "Mean Temp (°C)" in df.columns and "Production" in df.columns:
        temp_df = df.copy()
        temp_df["temp_range"] = pd.cut(
            temp_df["Mean Temp (°C)"],
            bins=[-30, -10, 0, 10, 20, 30],
            labels=["Very Cold", "Cold", "Mild", "Warm", "Hot"]
        )
        temp_prod = temp_df.groupby("temp_range", observed=False)["Production"].mean()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(temp_prod.index.astype(str), temp_prod.values)
        ax.set_title("Average Solar Production by Temperature Range")
        ax.set_xlabel("Temperature Range")
        ax.set_ylabel("Average Production")
        st.pyplot(fig)

    st.subheader("Average Solar Production by Wind Speed Range")
    if "wind_speed" in df.columns and "Production" in df.columns:
        wind_df = df.copy()
        wind_df["wind_range"] = pd.cut(
            wind_df["wind_speed"],
            bins=[0, 3, 5, 7, 10, 15],
            labels=["Very Low", "Low", "Moderate", "High", "Very High"]
        )
        wind_prod = wind_df.groupby("wind_range", observed=False)["Production"].mean()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(wind_prod.index.astype(str), wind_prod.values)
        ax.set_title("Average Solar Production by Wind Speed Range")
        ax.set_xlabel("Wind Speed Range")
        ax.set_ylabel("Average Production")
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

elif page == "Business Analysis":
    st.header("Business Analysis")

    if "combined_revenue" in business_df.columns:
        st.subheader("Best Time to Sell Solar Energy")
        best_months = business_df.groupby("month")["combined_revenue"].mean().sort_index()
        labels = [month_names[m-1] for m in best_months.index]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, best_months.values)
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Revenue")
        ax.set_title("Best Time to Sell Solar Energy")
        st.pyplot(fig)

        st.subheader("Solar Production and Revenue Trends Across the Year")
        monthly_prod = business_df.groupby("month")["combined_production"].mean().sort_index()
        monthly_rev = business_df.groupby("month")["combined_revenue"].mean().sort_index()
        labels_full = [month_names[m-1] for m in monthly_prod.index]

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(labels_full, monthly_prod.values, marker="o")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Average Production")

        ax2 = ax1.twinx()
        ax2.plot(labels_full, monthly_rev.values, marker="o")
        ax2.set_ylabel("Average Revenue")

        plt.title("Solar Production and Revenue Trends Across the Year")
        st.pyplot(fig)

elif page == "Modeling":
    st.header("Model Performance")

    if model_ready:
        st.write(f"**Model:** Random Forest Regressor")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

        st.subheader("Actual vs Predicted Solar Production")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, pred, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                linestyle="--")
        ax.set_xlabel("Actual Production")
        ax.set_ylabel("Predicted Production")
        ax.set_title("Actual vs Predicted Solar Production")
        ax.grid()
        st.pyplot(fig)
    else:
        st.error("Modeling columns are missing from merged_without_price.csv")

elif page == "XAI":
    st.header("Explainable AI (XAI)")

    if model_ready:
        st.subheader("Feature Importance")
        importance = pd.Series(rf.feature_importances_, index=available_features).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        importance.plot(kind="bar", ax=ax)
        ax.set_title("Feature Importance for Solar Production Prediction")
        ax.set_ylabel("Importance Score")
        st.pyplot(fig)

        st.subheader("Residual Plot")
        residuals = y_test - pred

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(pred, residuals)
        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Predicted Production")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        st.pyplot(fig)

        st.subheader("Key Insight")
        st.write("Solar radiation is the strongest driver of solar production, while temperature and seasonal features also contribute to prediction performance.")
    else:
        st.error("XAI plots cannot run because the modeling columns are missing.")
