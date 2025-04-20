
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load model + processor from Hugging Face
model_name = "motheecreator/vit-Facial-Expression-Recognition"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()

# Load dataset
df = pd.read_csv("ckextended.csv")

# Prepare result list
results = []

print("Running predictions on 100 samples...")

for idx, row in df.head(100).iterrows():  # Run on first 100 rows
    pixels = np.array(row['pixels'].split(), dtype='uint8').reshape(48, 48)
    img = Image.fromarray(pixels).convert("RGB")  # Convert to RGB

    # Use Hugging Face processor
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_label = model.config.id2label[pred_idx]
        confidence = probs[0][pred_idx].item()

    results.append({
        "original_label": row["emotion"],
        "predicted_emotion": pred_label,
        "confidence": round(confidence, 4)
    })

# Save output
out_df = pd.DataFrame(results)
out_df.to_csv("emotion_predictions.csv", index=False)
print("✅ Done! Results saved to emotion_predictions.csv")
