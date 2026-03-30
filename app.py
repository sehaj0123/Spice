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
visser = pd.read_csv("Visser_final_cleaned_filled (1).csv")
bissell = pd.read_csv("Bissell_inverters_production.csv")

# Fix dates
if "date" in business_df.columns:
    business_df["date"] = pd.to_datetime(business_df["date"])

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

visser["date"] = pd.to_datetime(visser["date"])
bissell["date"] = pd.to_datetime(bissell["date"])

visser["month"] = visser["date"].dt.month
bissell["month"] = bissell["date"].dt.month

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

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

if page == "Overview":
    st.header("Project Overview")
    st.write("""
    This project analyzes solar production using Visser, Bissell, weather,
    NASA solar radiation, and Alberta pool price data from AESO.
    """)
    st.write("""
    The app includes:
    - Exploratory Data Analysis (EDA)
    - Business Analysis
    - Machine Learning Modeling
    - Explainable AI (XAI)
    """)
    st.write("""
    The goal is to understand what affects solar production, compare site performance,
    estimate revenue, and explain how the prediction model works in a clear way for the client.
    """)

elif page == "EDA":
    st.header("Exploratory Data Analysis")

    st.subheader("Monthly Total Solar Production")
    monthly_total = business_df.groupby("month")["combined_production"].mean().sort_index()
    labels = [month_names[m - 1] for m in monthly_total.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, monthly_total.values)
    ax.set_title("Monthly Solar Production (Full Year)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Production")
    st.pyplot(fig)

    st.write("This chart shows full-year average solar production using the combined Visser and Bissell dataset.")

    st.subheader("Visser Monthly Production")
    visser_monthly = visser.groupby("month")["Production"].mean().sort_index()
    labels = [month_names[m - 1] for m in visser_monthly.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, visser_monthly.values)
    ax.set_title("Visser Monthly Production")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Production")
    st.pyplot(fig)

    st.write("This chart shows average monthly solar production for the Visser site only.")

    st.subheader("Bissell Monthly Production")
    bissell_monthly = bissell.groupby("month")["Bissell_total_filled"].mean().sort_index()
    labels = [month_names[m - 1] for m in bissell_monthly.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, bissell_monthly.values)
    ax.set_title("Bissell Monthly Production")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Production")
    st.pyplot(fig)

    st.write("This chart shows average monthly solar production for the Bissell site only.")

    st.subheader("Solar Radiation vs Production")
    if "solar_radiation" in df.columns and "Production" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(df["solar_radiation"], df["Production"], alpha=0.6)
        ax.set_xlabel("Solar Radiation")
        ax.set_ylabel("Solar Production")
        ax.set_title("Solar Radiation vs Solar Production")
        st.pyplot(fig)

        st.write("This scatter plot shows that higher solar radiation usually leads to higher solar production.")

    st.subheader("Average Solar Production by Temperature Range")
    if "Mean Temp (°C)" in df.columns and "Production" in df.columns:
        temp_df = df.copy()
        temp_df["temp_range"] = pd.cut(
            temp_df["Mean Temp (°C)"],
            bins=[-30, -10, 0, 10, 20, 30],
            labels=["-30 to -10", "-10 to 0", "0 to 10", "10 to 20", "20 to 30"]
        )
        temp_prod = temp_df.groupby("temp_range", observed=False)["Production"].mean()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(temp_prod.index.astype(str), temp_prod.values)
        ax.set_title("Average Solar Production by Temperature Range")
        ax.set_xlabel("Temperature Range (°C)")
        ax.set_ylabel("Average Production")
        st.pyplot(fig)

        st.write("This chart shows how solar production changes across temperature ranges. Production is lower in colder conditions and generally higher in warmer ranges.")

    st.subheader("Average Solar Production by Wind Speed Range")
    if "wind_speed" in df.columns and "Production" in df.columns:
        wind_df = df.copy()
        wind_df["wind_range"] = pd.cut(
            wind_df["wind_speed"],
            bins=[0, 3, 5, 7, 10, 15],
            labels=["0-3", "3-5", "5-7", "7-10", "10-15"]
        )
        wind_prod = wind_df.groupby("wind_range", observed=False)["Production"].mean()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(wind_prod.index.astype(str), wind_prod.values)
        ax.set_title("Average Solar Production by Wind Speed Range")
        ax.set_xlabel("Wind Speed Range")
        ax.set_ylabel("Average Production")
        st.pyplot(fig)

        st.write("This chart shows that wind speed has a weaker relationship with solar production compared with solar radiation and temperature.")

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

    st.write("This heatmap shows which variables have the strongest relationship with solar production. Solar radiation and temperature are among the strongest drivers.")

elif page == "Business Analysis":
    st.header("Business Analysis")

    if "combined_revenue" in business_df.columns:
        st.subheader("Best Time to Sell Solar Energy")
        best_months = business_df.groupby("month")["combined_revenue"].mean().sort_index()
        labels = [month_names[m - 1] for m in best_months.index]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, best_months.values)
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Revenue")
        ax.set_title("Best Time to Sell Solar Energy")
        st.pyplot(fig)

        st.write("This chart shows which months create the highest average solar revenue after combining production and pool price.")

        st.subheader("Solar Production and Revenue Trends Across the Year")
        monthly_prod = business_df.groupby("month")["combined_production"].mean().sort_index()
        monthly_rev = business_df.groupby("month")["combined_revenue"].mean().sort_index()
        labels_full = [month_names[m - 1] for m in monthly_prod.index]

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(labels_full, monthly_prod.values, marker="o")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Average Production")

        ax2 = ax1.twinx()
        ax2.plot(labels_full, monthly_rev.values, marker="o")
        ax2.set_ylabel("Average Revenue")

        plt.title("Solar Production and Revenue Trends Across the Year")
        st.pyplot(fig)

        st.write("This chart compares production and revenue through the year and shows that revenue depends on both solar output and electricity market price.")

elif page == "Modeling":
    st.header("Model Performance")

    if model_ready:
        st.write(f"**Model:** Random Forest Regressor")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

        st.write("The model uses environmental and seasonal features to predict solar production.")

        st.subheader("Actual vs Predicted Solar Production")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, pred, alpha=0.7)
        ax.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            linestyle="--"
        )
        ax.set_xlabel("Actual Production")
        ax.set_ylabel("Predicted Production")
        ax.set_title("Actual vs Predicted Solar Production")
        ax.grid()
        st.pyplot(fig)

        st.write("This plot shows how close the model predictions are to the actual production values. Points closer to the diagonal line indicate better prediction performance.")
    else:
        st.error("Modeling columns are missing from merged_without_price.csv")

elif page == "XAI":
    st.header("Explainable AI (XAI)")

    if model_ready:
        st.subheader("Feature Importance")
        importance = pd.Series(
            rf.feature_importances_,
            index=available_features
        ).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        importance.plot(kind="bar", ax=ax)
        ax.set_title("Feature Importance for Solar Production Prediction")
        ax.set_ylabel("Importance Score")
        st.pyplot(fig)

        st.write("This chart explains which features contributed the most to the solar production predictions.")

        st.subheader("Residual Plot")
        residuals = y_test - pred

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(pred, residuals)
        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Predicted Production")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        st.pyplot(fig)

        st.write("This plot shows model error. Residuals close to zero mean the model prediction is close to the actual value.")

        st.subheader("Key Insight")
        st.write("Solar radiation is the strongest driver of solar production, while temperature and seasonal variables also play an important role.")
    else:
        st.error("XAI plots cannot run because the modeling columns are missing.")
