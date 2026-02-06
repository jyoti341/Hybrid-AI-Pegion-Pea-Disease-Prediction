import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
LABELS_PATH = os.path.join(MODEL_DIR, 'hog_labels.npy')
LABELENCODER_PATH = os.path.join(MODEL_DIR, 'hog_labelencoder.pkl')

# Load labels from .npy file
labels = np.load(LABELS_PATH, allow_pickle=True)

# Create and fit LabelEncoder
le = LabelEncoder()
le.fit(labels)

# Save the LabelEncoder object as a pickle file
joblib.dump(le, LABELENCODER_PATH)

print(f"✅ LabelEncoder created and saved at: {LABELENCODER_PATH}")
