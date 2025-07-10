import os
import csv
from unsloth import FastVisionModel
from PIL import Image
import numpy as np
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model, tokenizer = FastVisionModel.from_pretrained(
    "0llheaven/Llama-3.2-11B-Vision-Radiology-mini",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)
FastVisionModel.for_inference(model)
model.to(device)

# Prediction function
def predict_radiology_description(image, instruction):
    try:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=1.0,
            min_p=0.1
        )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Data folder setup
data_root = "./data"
instruction = "You are an expert radiographer. What is the diagnosis in one word? If there is no illness, reply with 'normal'."

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
            diagnosis = predict_radiology_description(image, instruction)

            # Normalize: if no illness detected
            if "normal" in diagnosis.lower() or diagnosis.strip() == "":
                diagnosis = "normal"
            else:
                # Get just one word
                diagnosis = diagnosis.split()[0].strip().lower()

            results.append([image_name, subdir, diagnosis])
            print(f"[✓] {image_name} → {diagnosis}")
        except Exception as e:
            print(f"[X] Failed on {image_name}: {str(e)}")
            results.append([image_name, subdir, "error"])

# Save to CSV
csv_path = "radiology_diagnoses.csv"
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "directory", "model_diagnosis"])
    writer.writerows(results)

print(f"\n✅ Diagnoses saved to {csv_path}")
