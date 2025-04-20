import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page setup ---
st.set_page_config(page_title="Student Mood Tracker", layout="wide")
st.title("📚 Classroom Emotion Insights")

st.markdown(
    "This dashboard uses AI to recognize facial emotions from uploaded student images or live webcam input. "
    "Upload your own dataset, run predictions, or try it live!"
)

# --- File Upload Section ---
st.markdown("## 📂 Upload a Dataset (e.g. ckextended.csv)")
uploaded_file = st.file_uploader("Upload a CSV file with a 'pixels' column", type="csv")

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    if "pixels" not in raw_df.columns:
        st.error("The uploaded file must contain a 'pixels' column.")
    else:
        st.success("File uploaded successfully!")
        st.dataframe(raw_df.head())

        if st.button("🔍 Run Emotion Predictions on Uploaded Data"):
            from PIL import Image
            import numpy as np
            import torch
            from torchvision import transforms
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            model_name = "motheecreator/vit-Facial-Expression-Recognition"
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(model_name)
            model.eval()

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])

            st.info("Processing... please wait ⏳")
            results = []
            for idx, row in raw_df.iterrows():
                try:
                    pixels = np.array(row['pixels'].split(), dtype='uint8').reshape(48, 48)
                    image = transform(pixels)
                    image = image.unsqueeze(0)  # Batch
                    inputs = processor(images=image.squeeze().permute(1, 2, 0), return_tensors="pt")

                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        pred_idx = torch.argmax(probs, dim=1).item()
                        pred_label = model.config.id2label[pred_idx]
                        confidence = probs[0][pred_idx].item()

                    results.append({
                        "predicted_emotion": pred_label,
                        "confidence": round(confidence, 4)
                    })
                except Exception as e:
                    st.warning(f"⚠️ Could not process row {idx}: {e}")

            # Combine results
            predicted_df = pd.concat([raw_df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
            st.success("✅ Emotion predictions complete!")

            # --- Filters & Chart ---
            st.markdown("### 📊 Prediction Summary")

            conf_range = st.slider("Filter by Confidence", 0.0, 1.0, (0.0, 1.0), 0.01)
            emotions = predicted_df["predicted_emotion"].unique().tolist()
            selected_emotion = st.selectbox("Choose Emotion", ["All"] + sorted(emotions))

            filtered = predicted_df[(predicted_df["confidence"] >= conf_range[0]) & (predicted_df["confidence"] <= conf_range[1])]
            if selected_emotion != "All":
                filtered = filtered[filtered["predicted_emotion"] == selected_emotion]

            chart_data = (
                predicted_df["predicted_emotion"]
                .value_counts()
                .reset_index(name="count")
                .rename(columns={"index": "predicted_emotion"})
            )

            fig = px.bar(
                chart_data,
                x="predicted_emotion",
                y="count",
                color="predicted_emotion",
                title="How Often Each Emotion Was Detected"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.markdown("### 📋 Prediction Table")
            st.dataframe(filtered)

            # Download
            st.download_button(
                label="💾 Download Predictions as CSV",
                data=predicted_df.to_csv(index=False),
                file_name="emotion_predictions.csv",
                mime="text/csv"
            )

# --- Webcam Emotion Detection ---
st.markdown("---")
st.markdown("## 🎥 Try It Yourself – Webcam Emotion Detection")
use_camera = st.checkbox("📷 Turn on Webcam")

if use_camera:
    st.info("Click below to take a snapshot and we'll tell you what emotion we detect.")
    img_file_buffer = st.camera_input("📸 Capture your face")

    if img_file_buffer is not None:
        from PIL import Image
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        # Load model
        model_name = "motheecreator/vit-Facial-Expression-Recognition"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        model.eval()

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
