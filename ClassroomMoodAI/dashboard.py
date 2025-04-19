
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter

st.set_page_config(page_title="ClassroomMood AI Dashboard", layout="wide")

st.title("📊 ClassroomMood AI – Teacher Emotion Dashboard")

df = pd.read_csv("emotion_data.csv", parse_dates=["timestamp"])

st.sidebar.header("Filters")
selected_student = st.sidebar.selectbox("Select a student", ["All"] + sorted(df["student_id"].unique()))

if selected_student != "All":
    df = df[df["student_id"] == selected_student]

st.markdown("### 🧠 Emotion Trends Over Time")
fig = px.line(df, x="timestamp", y="emotion", color="student_id", markers=True)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### 📌 Current Engagement Summary")
latest_emotions = df.tail(10)["emotion"]
counts = Counter(latest_emotions)

alert = None
if (counts["bored"] + counts["neutral"] + counts["confused"] + counts["sad"]) >= 7:
    alert = "⚠️ High disengagement detected – consider changing pace or activity."

col1, col2 = st.columns(2)
with col1:
    st.metric("Happy", counts["happy"])
    st.metric("Neutral", counts["neutral"])
with col2:
    st.metric("Confused", counts["confused"])
    st.metric("Bored", counts["bored"])

if alert:
    st.warning(alert)

st.markdown("### 📂 Raw Data")
st.dataframe(df.tail(15))
