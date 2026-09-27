"""
Taxi Fare Predictor
--------------------
A Streamlit app that loads a pre-trained GradientBoostingRegressor model
(trained on a taxi trip pricing dataset) and predicts the trip price
based on trip details entered by the user.

Run locally with:
    streamlit run app.py

Make sure `gradient_boosting_taxi_model.pkl` is in the same folder as this
script (or update MODEL_PATH below).
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH = "gradient_boosting_taxi_model.pkl"

st.set_page_config(
    page_title="Taxi Fare Predictor",
    page_icon="🚕",
    layout="centered",
)


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(
        f"Couldn't find the model file `{MODEL_PATH}`. "
        "Place it in the same directory as this app."
    )
    st.stop()

FEATURE_ORDER = list(model.feature_names_in_)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🚕 Taxi Fare Predictor")
st.write(
    "Estimate a taxi trip price from trip details, using a Gradient "
    "Boosting model trained on historical taxi trip data."
)

st.divider()

# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
st.subheader("Trip details")

col1, col2 = st.columns(2)

with col1:
    trip_distance_km = st.number_input(
        "Trip distance (km)", min_value=0.0, max_value=200.0, value=10.0, step=0.5
    )
    passenger_count = st.number_input(
        "Passenger count", min_value=1, max_value=8, value=1, step=1
    )
    trip_duration_minutes = st.number_input(
        "Trip duration (minutes)", min_value=0.0, max_value=300.0, value=20.0, step=1.0
    )

with col2:
    base_fare = st.number_input(
        "Base fare ($)", min_value=0.0, max_value=50.0, value=3.0, step=0.5
    )
    per_km_rate = st.number_input(
        "Per-km rate ($/km)", min_value=0.0, max_value=20.0, value=1.5, step=0.1
    )
    per_minute_rate = st.number_input(
        "Per-minute rate ($/min)", min_value=0.0, max_value=10.0, value=0.3, step=0.05
    )

st.subheader("Conditions")

col3, col4 = st.columns(2)

with col3:
    time_of_day = st.selectbox(
        "Time of day", ["Afternoon", "Evening", "Morning", "Night"]
    )
    day_of_week = st.selectbox("Day of week", ["Weekday", "Weekend"])

with col4:
    traffic_conditions = st.selectbox(
        "Traffic conditions", ["High", "Low", "Medium"]
    )
    weather = st.selectbox("Weather", ["Clear", "Rain", "Snow"])

st.divider()

# --------------------------------------------------------------------------
# Build the feature row (matching the one-hot columns the model expects)
# --------------------------------------------------------------------------
row = {
    "Trip_Distance_km": trip_distance_km,
    "Passenger_Count": passenger_count,
    "Base_Fare": base_fare,
    "Per_Km_Rate": per_km_rate,
    "Per_Minute_Rate": per_minute_rate,
    "Trip_Duration_Minutes": trip_duration_minutes,
    "Time_of_Day_Evening": int(time_of_day == "Evening"),
    "Time_of_Day_Morning": int(time_of_day == "Morning"),
    "Time_of_Day_Night": int(time_of_day == "Night"),
    "Day_of_Week_Weekend": int(day_of_week == "Weekend"),
    "Traffic_Conditions_Low": int(traffic_conditions == "Low"),
    "Traffic_Conditions_Medium": int(traffic_conditions == "Medium"),
    "Weather_Rain": int(weather == "Rain"),
    "Weather_Snow": int(weather == "Snow"),
}

input_df = pd.DataFrame([row])[FEATURE_ORDER]

# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------
if st.button("Predict fare", type="primary", use_container_width=True):
    prediction = model.predict(input_df)[0]
    st.metric("Estimated trip price", f"${prediction:,.2f}")

with st.expander("See the model input row"):
    st.dataframe(input_df, use_container_width=True)

st.caption(
    "Model: scikit-learn GradientBoostingRegressor · "
    f"{model.n_estimators} estimators · learning rate {model.learning_rate}"
)