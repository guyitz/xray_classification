import os
import glob
import uuid
import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.multimodal import MultiModalPredictor

# Step 1: Collect image paths and labels
def collect_image_data(root_dir):
    image_paths = []
    labels = []
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                for img_path in glob.glob(os.path.join(class_dir, ext)):
                    image_paths.append(img_path)
                    labels.append(class_name)
    return pd.DataFrame({'image': image_paths, 'label': labels})

# Path to your dataset organized by class folders
data_dir = './data'  # This should contain subfolders like '01/', '02/', etc.
df = collect_image_data(data_dir)

# Step 2: Split into 80% train / 20% test
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

print(f"✅ Collected {len(df)} images: {len(train_df)} train / {len(test_df)} test")

# Step 3: Train the model using MultiModalPredictor
model_path = f"./autogluon_model_{uuid.uuid4().hex[:8]}"
predictor = MultiModalPredictor(label='label', path=model_path)

predictor.fit(
    train_data=train_df,
    time_limit=60,  # train for up to 10 minutes
)

# Step 4: Evaluate on test data
scores = predictor.evaluate(test_df, metrics=["accuracy"])
print(f"\n✅ Top-1 test accuracy: {scores['accuracy']:.3f}")
