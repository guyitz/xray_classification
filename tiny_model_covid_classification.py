import os
import csv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.nn import softmax

# Load the saved model
model = load_model("tiny_model_img_75x75.h5")

# Expected input image size
img_size = (75, 75)

# Map class indices to labels (based on training)
class_labels = {0: "NORMAL", 1: "COVID"}

# Folder containing image directories
data_root = "./data"
results = []

# Process images in each subfolder
for subdir in os.listdir(data_root):
    dir_path = os.path.join(data_root, subdir)
    if not os.path.isdir(dir_path):
        continue

    for image_name in os.listdir(dir_path):
        image_path = os.path.join(dir_path, image_name)
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        try:
            # Load and preprocess the image
            image = load_img(image_path, target_size=img_size)
            image_array = img_to_array(image) / 255.0  # Normalize to [0,1]
            image_array = np.expand_dims(image_array, axis=0)

            # Get logits and convert to probabilities
            logits = model.predict(image_array)
            probs = softmax(logits[0]).numpy()

            # Get top prediction
            top_idx = np.argmax(probs)
            label = class_labels[top_idx]
            score = round(probs[top_idx] * 100, 2)

            results.append([image_name, subdir, label, score])
            print(f"[✓] {image_name} → {label} ({score}%)")
        except Exception as e:
            print(f"[X] Failed on {image_name}: {str(e)}")
            results.append([image_name, subdir, "error", 0.0])

# Save results to CSV
csv_path = "covid_xray_predictions.csv"
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "directory", "classification", "model_percentage"])
    writer.writerows(results)

print(f"\n✅ Results saved to {csv_path}")