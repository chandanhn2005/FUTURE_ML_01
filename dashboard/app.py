import streamlit as st
import pandas as pd
import joblib
import sys
import os
import matplotlib.pyplot as plt

# -----------------------------
# Path Setup
# -----------------------------
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_PATH)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Forecast & Prediction Dashboard")
st.markdown("### Powered by XGBoost ML Model")

# -----------------------------
# Load Model & Features
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(ROOT_PATH, "models", "xgboost.pkl"))
    feature_columns = joblib.load(os.path.join(ROOT_PATH, "models", "xgb_feature_columns.pkl"))
    return model, feature_columns

model, feature_columns = load_model()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🔴 Real-Time Prediction", "📈 30-Day Forecast", "📊 Feature Importance"])

# ==================================================
# 🔴 TAB 1 — REAL TIME PREDICTION
# ==================================================
with tab1:

    st.subheader("Advanced Real-Time Sales Prediction")

    col1, col2 = st.columns(2)

    with col1:
        store = st.number_input("Store ID", min_value=1, step=1)
        date = st.date_input("Select Date")

        promo = 1 if st.selectbox("Is Promotion Running Today?", ["Yes", "No"]) == "Yes" else 0
        promo2 = 1 if st.selectbox("Is Long-Term Promotion Active?", ["Yes", "No"]) == "Yes" else 0
        school_holiday = 1 if st.selectbox("Is It a School Holiday?", ["Yes", "No"]) == "Yes" else 0

    with col2:
        open_store = 1 if st.selectbox("Store Status", ["Open", "Closed"]) == "Open" else 0

        store_type_display = st.selectbox(
            "Store Type",
            [
                "A → Basic store (small retail outlet)",
                "B → Extra store (medium sized)",
                "C → Extended store (large variety store)",
                "D → Premium store (high-end location)"
            ]
        )
        store_type = store_type_display[0]

        assortment_display = st.selectbox(
            "Assortment Type",
            [
                "A → Basic product range",
                "B → Medium product range",
                "C → Extended product range"
            ]
        )
        assortment = assortment_display[0]

        lag_1 = st.number_input("Yesterday Sales (Lag_1)", min_value=0.0)
        lag_7 = st.number_input("Sales 7 Days Ago (Lag_7)", min_value=0.0)

    # -----------------------------
    # 7-DAY AVERAGE OPTION
    # -----------------------------
    st.markdown("### 📊 7-Day Average Sales")

    avg_option = st.radio(
        "Choose input method:",
        ["Enter 7-Day Average Directly", "Calculate from 7 Days"]
    )

    if avg_option == "Enter 7-Day Average Directly":
        rolling_7 = st.number_input("7-Day Average Sales (Rolling_Mean_7)", min_value=0.0)
        daily_values = None
    else:
        st.markdown("#### 🧮 Enter Last 7 Days Sales")
        cols = st.columns(7)
        daily_values = []

        for i in range(7):
            value = cols[i].number_input(f"Day {i+1}", min_value=0.0, key=f"d{i}")
            daily_values.append(value)

        rolling_7 = sum(daily_values) / 7 if sum(daily_values) != 0 else 0
        st.info(f"Calculated 7-Day Average: {rolling_7:,.2f}")

    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button("Predict Sales", type="primary"):

        date = pd.to_datetime(date)

        input_data = {
            "Store": store,
            "DayOfWeek": date.weekday() + 1,
            "Open": open_store,
            "Promo": promo,
            "Promo2": promo2,
            "SchoolHoliday": school_holiday,
            "Year": date.year,
            "Month": date.month,
            "Day": date.day,
            "Weekday": date.weekday(),
            "Lag_1": lag_1,
            "Lag_7": lag_7,
            "Rolling_Mean_7": rolling_7
        }

        for stype in ["a", "b", "c", "d"]:
            input_data[f"StoreType_{stype}"] = 1 if store_type.lower() == stype else 0

        for atype in ["a", "b", "c"]:
            input_data[f"Assortment_{atype}"] = 1 if assortment.lower() == atype else 0

        df_input = pd.DataFrame([input_data])

        for col in feature_columns:
            if col not in df_input.columns:
                df_input[col] = 0

        df_input = df_input[feature_columns]

        prediction = model.predict(df_input)[0]

        # -----------------------------
        # KPI DISPLAY
        # -----------------------------
        st.markdown("## 📊 Prediction Result")

        st.metric("Predicted Sales", f"₹ {prediction:,.2f}")

        if lag_1 > 0:
            percent_change = ((prediction - lag_1) / lag_1) * 100
            st.metric(
                "Percentage Change vs Yesterday",
                f"{percent_change:.2f} %",
                delta=f"{percent_change:.2f}%"
            )

        # -----------------------------
        # GRAPH CONDITION
        # -----------------------------
        if avg_option == "Calculate from 7 Days" and daily_values is not None:

            st.markdown("### 📈 Sales Trend (Last 7 Days + Predicted)")

            sales_trend = daily_values + [prediction]
            labels = [f"D{i+1}" for i in range(7)] + ["Predicted"]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(labels, sales_trend, marker="o", linewidth=2)
            ax.scatter("Predicted", prediction, s=120)

            ax.set_ylabel("Sales")
            ax.set_title("7-Day Sales Trend with Prediction")

            plt.tight_layout()
            st.pyplot(fig)

        else:
            st.info("Trend graph available when using 'Calculate from 7 Days'.")

# ==================================================
# 📈 TAB 2 — 30 DAY FORECAST
# ==================================================
with tab2:

    st.subheader("Real-Time 30-Day Future Forecast")

    store_id = st.number_input("Enter Store ID for Forecast", min_value=1, step=1, key="forecast_store")

    if st.button("Generate 30-Day Forecast"):

        today = pd.Timestamp.today()
        future_dates = pd.date_range(start=today, periods=30)

        predictions = []

        for date in future_dates:

            input_data = {
                "Store": store_id,
                "DayOfWeek": date.weekday() + 1,
                "Open": 1,
                "Promo": 0,
                "SchoolHoliday": 0,
                "Year": date.year,
                "Month": date.month,
                "Day": date.day,
                "Weekday": date.weekday(),
                "Lag_1": 5000,
                "Lag_7": 5000,
                "Rolling_Mean_7": 5000
            }

            df_input = pd.DataFrame([input_data])

            for col in feature_columns:
                if col not in df_input.columns:
                    df_input[col] = 0

            df_input = df_input[feature_columns]
            prediction = model.predict(df_input)[0]
            predictions.append(prediction)

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted_Sales": predictions
        })

        st.dataframe(forecast_df)

        # -----------------------------
        # Calculate Statistics
        # -----------------------------
        avg_sales = forecast_df["Predicted_Sales"].mean()
        max_sales = forecast_df["Predicted_Sales"].max()
        min_sales = forecast_df["Predicted_Sales"].min()

        max_date = forecast_df.loc[forecast_df["Predicted_Sales"].idxmax(), "Date"]
        min_date = forecast_df.loc[forecast_df["Predicted_Sales"].idxmin(), "Date"]

        # Target (you can change this)
        target_sales = avg_sales * 1.10

        # -----------------------------
        # Professional Graph
        # -----------------------------
        fig, ax = plt.subplots(figsize=(12,6))

        # Main Forecast Line
        ax.plot(
            forecast_df["Date"],
            forecast_df["Predicted_Sales"],
            linewidth=3,
            marker="o",
            color="#1f77b4",
            label="Forecast Sales"
        )

        # Area Fill
        ax.fill_between(
            forecast_df["Date"],
            forecast_df["Predicted_Sales"],
            color="#1f77b4",
            alpha=0.15
        )

        # Average Line
        ax.axhline(
            avg_sales,
            linestyle="--",
            linewidth=2,
            color="green",
            label=f"Average Sales ({avg_sales:,.0f})"
        )

        # Target Line
        ax.axhline(
            target_sales,
            linestyle=":",
            linewidth=2,
            color="orange",
            label=f"Target Sales ({target_sales:,.0f})"
        )

        # Max Highlight
        ax.scatter(
            max_date,
            max_sales,
            color="green",
            s=120,
            label="Maximum Sales"
        )

        ax.text(
            max_date,
            max_sales,
            f" Max\n{max_sales:,.0f}",
            fontsize=9
        )

        # Min Highlight
        ax.scatter(
            min_date,
            min_sales,
            color="red",
            s=120,
            label="Minimum Sales"
        )

        ax.text(
            min_date,
            min_sales,
            f" Min\n{min_sales:,.0f}",
            fontsize=9
        )

        # Axis Labels
        ax.set_title("30-Day Sales Forecast", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Predicted Sales (₹)")

        # Grid
        ax.grid(True, linestyle="--", alpha=0.3)

        # Legend
        ax.legend()

        # Rotate Dates
        plt.xticks(rotation=45)

        plt.tight_layout()

        st.pyplot(fig)

# ==================================================
# 📊 TAB 3 — FEATURE IMPORTANCE
# ==================================================
with tab3:

    st.subheader("Feature Importance (XGBoost)")

    if hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(15)

        fig, ax = plt.subplots()
        ax.barh(importance_df["Feature"], importance_df["Importance"])
        ax.invert_yaxis()

        st.pyplot(fig)
        st.dataframe(importance_df)

    else:
        st.info("Feature importance not available.")