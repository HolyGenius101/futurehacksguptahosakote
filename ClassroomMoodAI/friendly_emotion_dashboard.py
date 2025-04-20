import streamlit as st
import pandas as pd
import plotly.express as px

# Page setup
st.set_page_config(page_title="Student Mood Tracker", layout="wide")
st.title("📚 Classroom Emotion Insights")

st.markdown(
    "This dashboard shows facial emotion predictions for students based on image analysis. "
    "You can use the filters below to focus on specific moods or confidence levels."
)

# Load data
df = pd.read_csv("emotion_predictions.csv")

# Sidebar filters
st.sidebar.header("🔍 Filters")
conf_range = st.sidebar.slider("Confidence Range", 0.0, 1.0, (0.0, 1.0), 0.01)
emotions = df["predicted_emotion"].unique().tolist()
selected_emotion = st.sidebar.selectbox("Choose Emotion", ["All"] + sorted(emotions))

# Filter logic
filtered_df = df[(df["confidence"] >= conf_range[0]) & (df["confidence"] <= conf_range[1])]
if selected_emotion != "All":
    filtered_df = filtered_df[filtered_df["predicted_emotion"] == selected_emotion]

# Main chart
st.markdown("### 😃 Emotion Distribution")
emotion_summary = (
    df["predicted_emotion"]
    .value_counts()
    .reset_index(name="count")
    .rename(columns={"index": "predicted_emotion"})
)
bar = px.bar(
    emotion_summary,
    x="predicted_emotion",
    y="count",
    color="predicted_emotion",
    title="How Often Each Emotion Was Detected"
)
st.plotly_chart(bar, use_container_width=True)

# Table
st.markdown("### 🧾 Prediction Details")
st.dataframe(filtered_df.drop(columns=["original_label"]) if "original_label" in filtered_df.columns else filtered_df)
