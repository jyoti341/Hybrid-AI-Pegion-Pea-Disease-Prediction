from flask import Flask, render_template, request, url_for
import os
import joblib
import cv2
import numpy as np

app = Flask(__name__)

# Load your trained SVM model
model = joblib.load('svm_model.pkl')

# Label mapping (exact order used in training)
disease_labels = {
    0: "Healthy Leaf 🌿",
    1: "Leaf Spot Disease 🍂",
    2: "Leaf Webber Disease 🕸️",
    3: "Sterility Mosaic Disease 🦠"
}

def extract_features(img):
    """Feature extraction EXACTLY SAME as training"""
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.flatten().reshape(1, -1)

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
    if image.filename == '':
        return "No file selected"

    # Save uploaded image
    upload_folder = os.path.join(app.root_path, 'static', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, image.filename)
    image.save(file_path)

    # Read image
    img = cv2.imread(file_path)

    # Extract features based on training
    features = extract_features(img)

    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = round(probabilities[prediction] * 100, 2)

    # Debug
    print("Raw Prediction:", prediction)

    # Map class index → class name
    result = disease_labels.get(prediction, "Unknown Disease")

    # URL for displaying image
    image_url = url_for('static', filename=f'uploads/{image.filename}')

    return render_template(
        'result.html',
        result=result,
        confidence=confidence,
        image_url=image_url
    )


if __name__ == '__main__':
    app.run(debug=True)
