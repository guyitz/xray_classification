import os
import csv
from PIL import Image
from transformers import pipeline

# Load model
classifier = pipeline(model="lxyuan/vit-xray-pneumonia-classification")

# Data folder setup
data_root = "./data"
results = []

# Walk through data folders
for subdir in os.listdir(data_root):
    dir_path = os.path.join(data_root, subdir)
    if not os.path.isdir(dir_path):
        continue

    for image_name in os.listdir(dir_path):
        image_path = os.path.join(dir_path, image_name)
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            prediction = classifier(image)

            # Get label and score of highest prediction
            top_pred = max(prediction, key=lambda x: x["score"])
            label = top_pred["label"]
            score = round(top_pred["score"] * 100, 2)  # Convert to percentage

            results.append([image_name, subdir, label, score])
            print(f"[✓] {image_name} → {label} ({score}%)")
        except Exception as e:
            print(f"[X] Failed on {image_name}: {str(e)}")
            results.append([image_name, subdir, "error", 0.0])

# Save to CSV
csv_path = "xray_classification_results.csv"
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "directory", "classification", "model_percentage"])
    writer.writerows(results)

print(f"\n✅ Results saved to {csv_path}")