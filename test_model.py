import os
import cv2
import joblib
import numpy as np

# Load model
model = joblib.load('svm_model.pkl')

# Categories (must match your training ones)
categories = ['Healthy', 'Leaf_Spot', 'Leaf_Webber', 'Sterility_Mosic']

# Path to your dataset (same as training)
data_path = r"E:\pegion_pea_crop\data"

# Automatically find the first image in the folders
img_path = None
for category in categories:
    folder = os.path.join(data_path, category)
    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder, file)
            break
    if img_path:
        break

if img_path is None:
    print("❌ No images found in your data folders.")
    exit()

print(f"📷 Using image: {img_path}")

# Read and preprocess
img = cv2.imread(img_path)
img = cv2.resize(img, (128, 128))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_flat = img.flatten().reshape(1, -1)

# Predict
prediction = model.predict(img_flat)[0]
print(f"🪴 Predicted Class: {categories[prediction]}")
