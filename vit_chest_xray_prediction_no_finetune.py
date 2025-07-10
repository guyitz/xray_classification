import os
import requests
import csv
from datetime import datetime

# === CONFIGURATION ===
DATA_ROOT = "./data"
PREDICT_URL = "http://localhost:8000/predict"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# === OUTPUT CSV ===
timestamp = datetime.now().strftime("%Y%m%d")
CSV_PATH = f"{timestamp}_xray_predictions.csv"

results = []

print(f"🚀 Sending images to: {PREDICT_URL}")
print(f"📁 Reading from: {DATA_ROOT}")
print(f"📄 Saving results to: {CSV_PATH}")

for folder in sorted(os.listdir(DATA_ROOT)):
    folder_path = os.path.join(DATA_ROOT, folder)
    if not os.path.isdir(folder_path):
        continue

    for image_file in sorted(os.listdir(folder_path)):
        if not image_file.lower().endswith(IMAGE_EXTENSIONS):
            continue

        image_path = os.path.join(folder_path, image_file)
        mime_type = "image/jpeg" if image_file.lower().endswith((".jpg", ".jpeg")) else "image/png"

        with open(image_path, "rb") as img:
            try:
                files = {
                    "image_file": (image_file, img, mime_type)
                }

                response = requests.post(PREDICT_URL, files=files)

                if response.status_code != 200:
                    raise Exception(f"Status {response.status_code}: {response.text}")

                data = response.json()
                predicted_label = data.get("predicted_class_label", "error")
                probabilities = data.get("probabilities", {})

                # Get confidence score for predicted label
                confidence = probabilities.get(predicted_label, 0.0)
                confidence_percent = round(confidence * 100, 1)  # e.g., 84.3%

                results.append([image_file, folder, predicted_label, confidence_percent])
                print(f"[✓] {image_file} ({folder}) → {predicted_label} ({confidence_percent}%)")

            except Exception as e:
                print(f"[X] Failed on {image_file}: {e}")
                results.append([image_file, folder, "error", 0])

# === WRITE TO CSV ===
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "directory", "predicted_label", "predicted_confidence_percent"])
    writer.writerows(results)

print(f"\n✅ Done! Results saved to {CSV_PATH}")