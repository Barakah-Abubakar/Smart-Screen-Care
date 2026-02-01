import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------------
# Page config
# -----------------------------------
st.set_page_config(
    page_title="Smart Screen Care",
    layout="centered"
)

st.title("👁️ Smart Screen Care")
st.caption("A personalized eye strain risk & recommendation system")
st.markdown(
    "**Disclaimer:** This is an educational screening tool, not a medical diagnosis."
)

st.divider()

# -----------------------------------
# Load & train model (inside app)
# -----------------------------------
data = pd.read_csv("eye_strain_dataset.csv")

FEATURES = [
    "age",
    "screen_time",
    "break_interval",
    "lighting",
    "screen_distance",
    "refractive_error"
]

X = data[FEATURES]
y = data["eye_strain_score"]

model = LinearRegression()
model.fit(X, y)

# -----------------------------------
# Recommendation Logic
# -----------------------------------
def generate_recommendations(score, refractive_error):
    recommendations = []

    if score < 3:
        risk = "Low"
    elif score < 6:
        risk = "Moderate"
    else:
        risk = "High"

    if risk == "Low":
        recommendations.extend([
            "Maintain your current screen habits.",
            "Use the 20-20-20 rule to prevent fatigue.",
            "Blink consciously during screen use."
        ])

    elif risk == "Moderate":
        recommendations.extend([
            "Reduce continuous screen exposure.",
            "Take breaks every 20–30 minutes.",
            "Optimize screen brightness and room lighting."
        ])

    else:
        recommendations.extend([
            "Limit non-essential screen time.",
            "Take breaks every 15–20 minutes.",
            "Avoid screens in dim lighting.",
            "Consider professional eye evaluation."
        ])

    if refractive_error != "None":
        recommendations.extend([
            "Your refractive error increases sensitivity to screen strain.",
            "Ensure your corrective lenses are up-to-date.",
            "Use screen-specific or anti-glare lenses if available.",
            "Increase break frequency beyond standard recommendations."
        ])

    return risk, recommendations

# -----------------------------------
# User Inputs (NO SLIDERS)
# -----------------------------------
st.subheader("🧠 Daily Screen & Eye Health Check")

age = st.number_input("Age", min_value=10, max_value=80, value=22)

screen_time = st.number_input(
    "Daily screen time (hours)",
    min_value=0.0,
    max_value=24.0,
    value=6.0
)

break_interval = st.number_input(
    "Average break interval (minutes)",
    min_value=5,
    max_value=120,
    value=30
)

lighting = st.selectbox(
    "Primary screen lighting condition",
    ["Poor", "Moderate", "Good"]
)

screen_distance = st.number_input(
    "Average screen distance (cm)",
    min_value=20,
    max_value=100,
    value=40
)

refractive_error = st.selectbox(
    "Do you have a refractive error?",
    ["None", "Myopia", "Hyperopia", "Astigmatism"]
)

# Encoding maps
lighting_map = {"Poor": 0, "Moderate": 1, "Good": 2}
refractive_map = {
    "None": 0,
    "Myopia": 1,
    "Hyperopia": 2,
    "Astigmatism": 3
}

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("Predict Eye Strain"):
    user_input = pd.DataFrame([{
        "age": age,
        "screen_time": screen_time,
        "break_interval": break_interval,
        "lighting": lighting_map[lighting],
        "screen_distance": screen_distance,
        "refractive_error": refractive_map[refractive_error]
    }])

    score = model.predict(user_input)[0]
    risk, advice = generate_recommendations(score, refractive_error)

    st.divider()
    st.subheader("🔍 Eye Strain Assessment")

    st.metric(
        label="Predicted Eye Strain Score",
        value=f"{score:.1f} / 10"
    )

    st.write(f"**Risk Level:** {risk}")

    st.subheader("📝 Personalized Recommendations")
    for rec in advice:
        st.write("•", rec)

    st.info(
        "Consistent eye-friendly habits today can prevent long-term vision problems."
    )
