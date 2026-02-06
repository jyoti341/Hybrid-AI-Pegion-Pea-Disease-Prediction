import numpy as np
import os

# Make sure folder exists
os.makedirs("model", exist_ok=True)

# 🔹 Labels in the SAME ORDER used during training
# (check how you trained your model — adjust order if needed)
labels = np.array([
    "healthy",
    "leaf spot",
    "leaf webber",
    "sterility mosaic"
])

# Save the labels file
np.save("model/hog_labels.npy", labels)

print("✅ Corrected labels file saved successfully!")
