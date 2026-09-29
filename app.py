from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "gradient_boosting_tuned.pkl"

# Categories. The first option in each list is the "baseline" that the model
# was trained without (one-hot encoding with drop_first=True), so it maps to
# all-zero dummy columns.
TIME_OF_DAY = ["Afternoon", "Morning", "Evening", "Night"]
DAY_OF_WEEK = ["Weekday", "Weekend"]
TRAFFIC = ["High", "Medium", "Low"]
WEATHER = ["Clear", "Rain", "Snow"]


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(MODEL_PATH)


def build_features(
    distance_km,
    passengers,
    base_fare,
    per_km_rate,
    per_minute_rate,
    duration_min,
    time_of_day,
    day_of_week,
    traffic,
    weather,
    feature_order,
):
    """Return a one-row DataFrame with the exact columns the model expects."""
    row = {
        "Trip_Distance_km": float(distance_km),
        "Passenger_Count": float(passengers),
        "Base_Fare": float(base_fare),
        "Per_Km_Rate": float(per_km_rate),
        "Per_Minute_Rate": float(per_minute_rate),
        "Trip_Duration_Minutes": float(duration_min),
        "Time_of_Day_Evening": int(time_of_day == "Evening"),
        "Time_of_Day_Morning": int(time_of_day == "Morning"),
        "Time_of_Day_Night": int(time_of_day == "Night"),
        "Day_of_Week_Weekend": int(day_of_week == "Weekend"),
        "Traffic_Conditions_Low": int(traffic == "Low"),
        "Traffic_Conditions_Medium": int(traffic == "Medium"),
        "Weather_Rain": int(weather == "Rain"),
        "Weather_Snow": int(weather == "Snow"),
    }
    return pd.DataFrame([row])[list(feature_order)]


def main():
    st.set_page_config(page_title="Trip Price Predictor", page_icon="🚕", layout="centered")
    st.title("🚕 Trip Price Predictor")
    st.caption("Predict the price of a taxi trip using a tuned Gradient Boosting model.")

    try:
        model = load_model()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load the model file `{MODEL_PATH.name}`.")
        st.exception(exc)
        st.stop()

    feature_order = list(getattr(model, "feature_names_in_", []))
    if not feature_order:
        st.error("The model does not contain feature names, so inputs cannot be aligned.")
        st.stop()

    st.subheader("Trip details")
    col1, col2 = st.columns(2)
    with col1:
        distance_km = st.number_input("Trip distance (km)", 0.1, 500.0, 15.0, 0.5)
        duration_min = st.number_input("Trip duration (minutes)", 1.0, 600.0, 30.0, 1.0)
        passengers = st.number_input("Passenger count", 1, 8, 1, 1)
        time_of_day = st.selectbox("Time of day", TIME_OF_DAY)
        day_of_week = st.selectbox("Day of week", DAY_OF_WEEK)
    with col2:
        base_fare = st.number_input("Base fare", 0.0, 100.0, 3.5, 0.1)
        per_km_rate = st.number_input("Per-km rate", 0.0, 20.0, 1.2, 0.05)
        per_minute_rate = st.number_input("Per-minute rate", 0.0, 10.0, 0.3, 0.05)
        traffic = st.selectbox("Traffic conditions", TRAFFIC)
        weather = st.selectbox("Weather", WEATHER)

    if st.button("Predict price", type="primary", use_container_width=True):
        try:
            X = build_features(
                distance_km, passengers, base_fare, per_km_rate, per_minute_rate,
                duration_min, time_of_day, day_of_week, traffic, weather, feature_order,
            )
            prediction = max(float(model.predict(X)[0]), 0.0)
        except Exception as exc:  # noqa: BLE001
            st.error("Prediction failed.")
            st.exception(exc)
        else:
            st.success(f"Estimated trip price: **{prediction:,.2f}**")
            with st.expander("Show model input"):
                st.dataframe(X.T.rename(columns={0: "value"}))


if __name__ == "__main__":
    main()
