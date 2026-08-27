"""
Air Quality Forecaster
NeuroFive ML Track, Capstone Project

Predicts PM2.5 concentration from weather conditions using a model
trained on five years of hourly Beijing air quality data.
"""

import streamlit as st
import pandas as pd
import joblib
import datetime

st.set_page_config(
    page_title="Air Quality Forecaster",
    page_icon="🌫️",
    layout="centered"
)

MODEL_PATH = "aqi_model.joblib"

FEATURE_COLUMNS = [
    "TEMP", "DEWP", "PRES", "Iws", "Is", "Ir",
    "hour", "month", "cbwd",
    "heating_season", "temp_dewp_spread", "is_night", "season"
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error(f"Could not find {MODEL_PATH}. Make sure it is committed to the repository.")
    st.stop()
except Exception as error:
    st.error(
        "The model could not be loaded. This is usually a library version mismatch. "
        "Check that requirements.txt pins the same scikit-learn version used in training."
    )
    st.exception(error)
    st.stop()


def categorise_pm25(value):
    """Map a PM2.5 concentration to an air quality band and health message."""
    if value <= 12:
        return "Good", "#2ecc71", "Air quality is satisfactory. No precautions needed."
    if value <= 35.4:
        return "Moderate", "#f1c40f", "Acceptable, though unusually sensitive people may want to limit long outdoor exertion."
    if value <= 55.4:
        return "Unhealthy for Sensitive Groups", "#e67e22", "Children, older adults and people with asthma or heart conditions should reduce prolonged outdoor exertion."
    if value <= 150.4:
        return "Unhealthy", "#e74c3c", "Everyone may begin to feel effects. Sensitive groups should avoid prolonged outdoor exertion."
    if value <= 250.4:
        return "Very Unhealthy", "#9b59b6", "Health warnings. Everyone should avoid outdoor exertion and keep windows closed."
    return "Hazardous", "#7f1d1d", "Emergency conditions. Everyone should remain indoors and use air filtration if available."


def build_features(date, hour, temp, dewp, pres, wind_dir, wind_speed, rain_hours, snow_hours):
    """Recreate the engineered features exactly as the training notebook did."""
    month = date.month
    day = date.day

    heating_season = int(
        (month == 11 and day >= 15)
        or month in (12, 1, 2)
        or (month == 3 and day <= 15)
    )

    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"
    }

    return pd.DataFrame([{
        "TEMP": float(temp),
        "DEWP": float(dewp),
        "PRES": float(pres),
        "Iws": float(wind_speed),
        "Is": float(snow_hours),
        "Ir": float(rain_hours),
        "hour": int(hour),
        "month": int(month),
        "cbwd": wind_dir,
        "heating_season": heating_season,
        "temp_dewp_spread": float(temp) - float(dewp),
        "is_night": int(hour < 6 or hour >= 20),
        "season": season_map[month]
    }])


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.title("🌫️ Air Quality Forecaster")

st.write(
    "Estimate PM2.5 concentration from weather conditions. PM2.5 refers to airborne "
    "particles under 2.5 micrometres across, small enough to reach deep into the lungs "
    "and enter the bloodstream, which is why it is the pollutant most closely linked to "
    "respiratory and cardiovascular harm."
)

st.divider()

# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------

st.subheader("Conditions")

col1, col2 = st.columns(2)

with col1:
    date = st.date_input("Date", value=datetime.date(2014, 1, 15))
    hour = st.slider("Hour of day", 0, 23, 14)
    temp = st.slider("Temperature (°C)", -20.0, 45.0, 5.0, step=0.5)
    dewp = st.slider("Dew point (°C)", -40.0, 30.0, -5.0, step=0.5)

with col2:
    pres = st.slider("Pressure (hPa)", 990.0, 1050.0, 1020.0, step=0.5)
    wind_dir = st.selectbox(
        "Wind direction",
        options=["NW", "NE", "SE", "cv"],
        format_func=lambda x: {
            "NW": "North West", "NE": "North East",
            "SE": "South East", "cv": "Calm and variable"
        }[x]
    )
    wind_speed = st.slider(
        "Cumulated wind speed (m/s)", 0.0, 300.0, 10.0, step=1.0,
        help="Running total of wind speed. Sustained wind clears pollution, so high values usually mean cleaner air."
    )
    rain_hours = st.slider("Cumulated hours of rain", 0, 36, 0)
    snow_hours = st.slider("Cumulated hours of snow", 0, 36, 0)

if dewp > temp:
    st.warning(
        "Dew point is above temperature, which is physically impossible. "
        "The prediction below is unreliable."
    )

st.divider()

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------

if st.button("Predict Air Quality", type="primary", width='stretch'):

    features = build_features(
        date, hour, temp, dewp, pres, wind_dir, wind_speed, rain_hours, snow_hours
    )

    prediction = float(model.predict(features[FEATURE_COLUMNS])[0])
    prediction = max(prediction, 0.0)

    band, colour, advice = categorise_pm25(prediction)

    st.subheader("Prediction")

    st.markdown(
        f"<div style='background-color:{colour};padding:20px;border-radius:10px;text-align:center'>"
        f"<h1 style='color:white;margin:0'>{prediction:.1f} µg/m³</h1>"
        f"<h3 style='color:white;margin:0'>{band}</h3>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.write("")
    st.info(advice)

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Predicted PM2.5", f"{prediction:.1f} µg/m³")
    metric_col2.metric(
        "Against WHO 24h guideline (15 µg/m³)",
        f"{prediction / 15:.1f}x"
    )

    if features["heating_season"].iloc[0] == 1:
        st.caption(
            "This date falls in the winter heating season, when coal fired district "
            "heating raises baseline pollution."
        )

    with st.expander("What the model received"):
        st.dataframe(features[FEATURE_COLUMNS], width='stretch')

st.divider()

with st.expander("About this model and its limits"):
    st.markdown(
        """
        **Data:** 41,757 hourly readings from Beijing, 2010 to 2014, pairing US Embassy
        PM2.5 measurements with meteorological data from Beijing Capital International Airport.

        **Evaluation:** trained on 2010 to 2013 and tested on 2014, so the reported
        performance reflects predicting a year the model had never seen. A random
        train/test split would have produced a far more flattering number by letting the
        model learn from hours adjacent to the ones it was tested on.

        **What it does not do.** This model predicts pollution from *weather alone*. It has
        no knowledge of traffic volume, industrial output, construction, or policy changes,
        all of which matter. Weather sets the conditions in which pollution accumulates or
        disperses; it does not create the pollution.

        **Geographic limits.** It learned Beijing's specific mix of emission sources and
        terrain. Applying it unchanged to another city would be unsound, though the same
        method would work anywhere with comparable historical data.

        **Category bands** follow US EPA PM2.5 breakpoints, which are defined for 24 hour
        averages while this model predicts a single hour. Treat the band as indicative.

        Built for the NeuroFive ML Track capstone project.
        """
    )
