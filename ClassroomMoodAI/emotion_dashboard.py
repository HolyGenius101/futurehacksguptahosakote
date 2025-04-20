
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter

st.set_page_config(page_title="Classroom Emotion Recognition", layout="wide")
st.title("🧠 Emotion Recognition Results Dashboard")

# Load predictions
df = pd.read_csv("emotion_predictions.csv")

# Overview
st.markdown("### 📊 Summary of Predictions")
emotion_counts = df["predicted_emotion"].value_counts().reset_index()
emotion_counts.columns = ["Emotion", "Count"]
fig = px.bar(emotion_counts, x="Emotion", y="Count", color="Emotion", title="Predicted Emotion Distribution")
st.plotly_chart(fig, use_container_width=True)

# Confidence slider
conf_range = st.slider("Filter by Confidence Score", 0.0, 1.0, (0.0, 1.0), 0.01)
filtered_df = df[(df["confidence"] >= conf_range[0]) & (df["confidence"] <= conf_range[1])]

# Emotion filter
emotions = df["predicted_emotion"].unique().tolist()
selected_emotion = st.selectbox("Filter by Emotion", ["All"] + emotions)
if selected_emotion != "All":
    filtered_df = filtered_df[filtered_df["predicted_emotion"] == selected_emotion]

# Show filtered table
st.markdown("### 📋 Filtered Prediction Results")
st.dataframe(filtered_df.reset_index(drop=True))
