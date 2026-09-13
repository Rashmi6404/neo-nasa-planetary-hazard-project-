import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="NASA NEO Planetary Hazard Intelligence",
    page_icon="☄️",
    layout="wide"
)

st.title("☄️ NASA Near-Earth Objects (NEO) Hazard Intelligence")
st.markdown("Interactive planetary hazard surveillance using NASA NEO data and machine learning.")

@st.cache_data(ttl=3600)
def get_neo_data():
    url = "https://api.nasa.gov/neo/rest/v1/feed"
    params = {"api_key": "DEMO_KEY"}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    rows = []
    for date, objects in data.get("near_earth_objects", {}).items():
        for obj in objects:
            diameter = obj.get("estimated_diameter", {}).get("kilometers", {})
            approach = obj.get("close_approach_data", [])
            approach = approach[0] if approach else {}

            rows.append({
                "Date": date,
                "Object ID": obj.get("id"),
                "Name": obj.get("name"),
                "Magnitude H": obj.get("absolute_magnitude_h"),
                "Diameter Min (km)": diameter.get("estimated_diameter_min"),
                "Diameter Max (km)": diameter.get("estimated_diameter_max"),
                "Min Distance(au)": (
                    approach.get("miss_distance", {}).get("astronomical")
                ),
                "Relative Velocity(km/s)": (
                    approach.get("relative_velocity", {}).get("kilometers_per_second")
                ),
                "Orbiting Body": approach.get("orbiting_body"),
                "NASA PHA Flag": int(obj.get("is_potentially_hazardous_asteroid", False))
            })

    df = pd.DataFrame(rows)

    numeric_cols = [
        "Magnitude H",
        "Diameter Min (km)",
        "Diameter Max (km)",
        "Min Distance(au)",
        "Relative Velocity(km/s)"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year

    df["is_hazardous"] = (
        (df["Min Distance(au)"] <= 0.05) &
        (df["Magnitude H"] <= 22.0)
    ).astype(int)

    return df


@st.cache_resource
def train_model(df):
    features = [
        "Min Distance(au)",
        "Relative Velocity(km/s)",
        "Magnitude H",
        "Diameter Min (km)",
        "Diameter Max (km)"
    ]

    X = df[features]
    y = df["is_hazardous"]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(X, y)
    return model, features


try:
    df = get_neo_data()
    model, features = train_model(df)
except Exception as e:
    st.error("Unable to retrieve NASA NEO data right now.")
    st.info("Please refresh the app and try again.")
    st.stop()

# ---------------- KPIs ----------------
c1, c2, c3, c4 = st.columns(4)

c1.metric("NEO Records", f"{len(df):,}")
c2.metric("Unique Objects", f"{df['Object ID'].nunique():,}")
c3.metric(
    "Hazardous",
    f"{df['is_hazardous'].sum():,}"
)
c4.metric(
    "Avg Velocity",
    f"{df['Relative Velocity(km/s)'].mean():.2f} km/s"
)

st.divider()

tab1, tab2, tab3 = st.tabs([
    "📊 Analytics",
    "🔮 Hazard Classifier",
    "📋 NEO Data"
])

# ---------------- ANALYTICS ----------------
with tab1:
    left, right = st.columns(2)

    with left:
        st.subheader("NEO Discoveries by Date")
        daily = df.groupby("Date").size().reset_index(name="Objects")

        fig = px.line(
            daily,
            x="Date",
            y="Objects",
            title="Near-Earth Object Records"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Distance vs Relative Velocity")

        plot_df = df.dropna(
            subset=["Min Distance(au)", "Relative Velocity(km/s)"]
        )

        fig2 = px.scatter(
            plot_df,
            x="Min Distance(au)",
            y="Relative Velocity(km/s)",
            color="is_hazardous",
            hover_data=["Name", "Magnitude H"],
            title="Proximity vs Velocity"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Hazard Classification")

    counts = df["is_hazardous"].value_counts().rename(
        {0: "Non-Hazardous", 1: "Hazardous"}
    ).reset_index()

    counts.columns = ["Classification", "Count"]

    fig3 = px.bar(
        counts,
        x="Classification",
        y="Count",
        title="Hazardous vs Non-Hazardous NEOs"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ---------------- CLASSIFIER ----------------
with tab2:
    st.subheader("🔮 Evaluate NEO Hazard")

    st.write(
        "Enter trajectory and physical characteristics. "
        "The Random Forest model estimates the hazard probability."
    )

    col1, col2 = st.columns(2)

    with col1:
        distance = st.number_input(
            "Minimum Distance (au)",
            min_value=0.0001,
            value=0.045,
            step=0.005
        )

        velocity = st.number_input(
            "Relative Velocity (km/s)",
            min_value=0.0,
            value=28.5,
            step=1.0
        )

        magnitude = st.number_input(
            "Absolute Magnitude H",
            value=20.0,
            step=0.5
        )

    with col2:
        diameter_min = st.number_input(
            "Minimum Diameter (km)",
            min_value=0.0,
            value=0.1,
            step=0.01
        )

        diameter_max = st.number_input(
            "Maximum Diameter (km)",
            min_value=0.0,
            value=0.2,
            step=0.01
        )

    if st.button("🚨 Evaluate Hazard", type="primary"):
        input_df = pd.DataFrame([{
            "Min Distance(au)": distance,
            "Relative Velocity(km/s)": velocity,
            "Magnitude H": magnitude,
            "Diameter Min (km)": diameter_min,
            "Diameter Max (km)": diameter_max
        }])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(
                f"⚠️ HAZARD ALERT — Potentially Hazardous Object\n\n"
                f"Model probability: {probability * 100:.1f}%"
            )
        else:
            st.success(
                f"✅ NON-HAZARDOUS — Lower predicted risk\n\n"
                f"Hazard probability: {probability * 100:.1f}%"
            )

# ---------------- DATA ----------------
with tab3:
    st.subheader("NASA NEO Dataset")
    st.dataframe(
        df.sort_values("Min Distance(au)").reset_index(drop=True),
        use_container_width=True
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download NEO Data",
        csv,
        "nasa_neo_data.csv",
        "text/csv"
    )

st.caption(
    "Data source: NASA Near Earth Object Web Service (NeoWs). "
    "Hazard rule follows the project definition: "
    "minimum distance ≤ 0.05 au and absolute magnitude H ≤ 22."
)
