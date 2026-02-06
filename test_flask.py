from flask import Flask, render_template, request, url_for
import os
import joblib
import cv2
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load('svm_model.pkl')

# HOG descriptor (same for training + prediction)
hog = cv2.HOGDescriptor()

# -----------------------------
# FEATURE EXTRACTION FUNCTION
# -----------------------------
def extract_features(img):
    img = cv2.resize(img, (128, 128))

    # HSV features
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [8, 8, 8], [0, 180, 0, 256, 0, 256]
    )
    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()

    # HOG features
    h = hog.compute(img).flatten()

    # Combine features
    return np.hstack([h, hsv_hist])


# -----------------------------
# ROUTES
# -----------------------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect')
def detect():
    return render_template('disease_detection.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded"

    image = request.files['image']

    # Save uploaded image
    upload_folder = os.path.join(app.root_path, 'static', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, image.filename)
    image.save(file_path)

    # Read and extract features
    img = cv2.imread(file_path)
    features = extract_features(img).reshape(1, -1)

    # Prediction
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = round(probabilities[prediction] * 100, 2)

    disease_labels = {
        0: "Healthy Leaf 🌿",
        1: "Leaf Spot Disease 🍂",
        2: "Leaf Webber Disease 🕸️",
        3: "Sterility Mosaic Disease 🦠"
    }

    result = disease_labels.get(prediction, "Unknown Disease")

    # Image URL for HTML display
    image_url = url_for('static', filename=f'uploads/{image.filename}')

    return render_template(
        'result.html',
        result=result,
        image_url=image_url,
        confidence=confidence
    )


if __name__ == '__main__':
    app.run(debug=True)
