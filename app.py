import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(
    page_title="NASA NEO Planetary Hazard Intelligence",
    page_icon="☄️",
    layout="wide",
)

@st.cache_data
def load_data():
    return pd.read_parquet("neo_cleaned_data.parquet")

@st.cache_resource
def load_model():
    model = joblib.load("pha_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features

df = load_data()
model, features = load_model()

st.title("☄️ NASA Near-Earth Objects (NEO) Hazard Intelligence")
st.markdown("Automated risk classification and trajectory surveillance dashboard.")

# Top KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Close Approaches", f"{len(df):,}")
c2.metric("Unique Asteroids", f"{df['Object ID'].nunique():,}")
c3.metric(
    "Hazardous (PHA)",
    f"{df['is_hazardous'].sum():,} ({df['is_hazardous'].mean()*100:.1f}%)"
)
c4.metric(
    "Avg Velocity",
    f"{df['Relative Velocity(km/s)'].mean():.2f} km/s"
)

st.divider()

tab1, tab2, tab3 = st.tabs(
    ["📊 Analytics", "🔮 Real-Time Classifier", "🔍 Model Explainability (SHAP)"]
)

with tab1:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Approach Frequency by Year")
        yearly_counts = df.groupby("Year").size().reset_index(name="Approaches")
        fig_year = px.line(
            yearly_counts,
            x="Year",
            y="Approaches",
            title="Approaches (1900–2025)",
            color_discrete_sequence=["#1f77b4"],
        )
        st.plotly_chart(fig_year, use_container_width=True)

    with col_right:
        st.subheader("Proximity vs Velocity Distribution")
        sample_df = df.sample(min(3000, len(df)), random_state=42)

        fig_scatter = px.scatter(
            sample_df,
            x="Min Distance(au)",
            y="Relative Velocity(km/s)",
            color=sample_df["is_hazardous"].map(
                {1: "Hazardous (PHA)", 0: "Safe"}
            ),
            color_discrete_map={
                "Hazardous (PHA)": "#d62728",
                "Safe": "#2ca02c",
            },
            title="Min Distance (au) vs Velocity (km/s)",
            opacity=0.6,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.subheader("Asteroid Hazard Prediction (Leak-Free Model)")
    st.write(
        "Input non-leaked trajectory metrics to evaluate collision hazard level:"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        nom_dist = st.number_input(
            "Nominal Distance (au)", value=0.045, step=0.005
        )
        max_dist = st.number_input(
            "Max Distance (au)", value=0.052, step=0.005
        )

    with col2:
        rel_vel = st.number_input(
            "Relative Velocity (km/s)", value=28.5, step=1.0
        )
        inf_vel = st.number_input(
            "Infinity Velocity (km/s)", value=28.3, step=1.0
        )

    with col3:
        time_unc = st.number_input(
            "Time Uncertainty (min)", value=15.0, step=1.0
        )
        orbit_id = st.number_input("Orbit ID", value=16, step=1)
        julian_date = st.number_input(
            "Julian Date", value=2450000.0, step=100.0
        )

    if st.button("Evaluate Hazard"):
        input_data = pd.DataFrame(
            [[
                nom_dist,
                max_dist,
                rel_vel,
                inf_vel,
                time_unc,
                orbit_id,
                julian_date,
            ]],
            columns=features,
        )

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if pred == 1:
            st.error(
                f"⚠️ **HAZARD ALERT:** High Risk (PHA)! "
                f"Model Probability: {prob*100:.1f}%"
            )
        else:
            st.success(
                f"✅ **SAFE:** Non-Hazardous Object. "
                f"Safety Probability: {(1-prob)*100:.1f}%"
            )

with tab3:
    st.subheader("Feature Importance (SHAP TreeExplainer)")
    st.image(
        "shap_summary.png",
        caption="Global SHAP Attribution Summary",
    )
