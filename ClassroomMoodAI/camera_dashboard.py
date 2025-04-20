import streamlit as st
import pandas as pd
import plotly.express as px

# Page setup
st.set_page_config(page_title="Student Mood Tracker", layout="wide")
st.title("📚 Classroom Emotion Insights")

st.markdown(
    "This dashboard shows facial emotion predictions for students based on image analysis. "
    "Use the filters below to explore results — or scroll down to try our live webcam emotion detection!"
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

# Chart
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

# Webcam Emotion Detection
st.markdown("---")
st.markdown("### 🎥 Try It Yourself – Webcam Emotion Detection")

use_camera = st.checkbox("📷 Turn on Webcam")

if use_camera:
    st.info("Click below to take a snapshot and we'll tell you what emotion we detect.")
    img_file_buffer = st.camera_input("📸 Capture your face")

    if img_file_buffer is not None:
        from PIL import Image
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        # Load model + processor
        model_name = "motheecreator/vit-Facial-Expression-Recognition"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        model.eval()

        # Process the image
        img = Image.open(img_file_buffer).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = probs.argmax().item()
            label = model.config.id2label[pred_idx]
            confidence = probs[0][pred_idx].item()

        st.success(f"🧠 Detected Emotion: {label} ({confidence:.2%} confidence)")
        st.image(img, caption="Your Captured Image", use_container_width=True)
else:
    st.info("Webcam is currently turned off. Turn it on above to test real-time emotion detection.")
