from flask import Flask, render_template, request, url_for
import os
import joblib
import cv2
import numpy as np

app = Flask(__name__)

# Load your trained SVM model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'svm_model.pkl')

model = joblib.load(model_path)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Disease detection page
@app.route('/detect')
def detect():
    return render_template('disease_detection.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded"

    image = request.files['image']
    if image.filename == '':
        return "No file selected"

    # Save the uploaded image to the app's static/uploads folder (filesystem path)
    upload_folder = os.path.join(app.root_path, 'static', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, image.filename)
    image.save(file_path)

    # Read and preprocess image (use filesystem path)
    img = cv2.imread(file_path)
    img = cv2.resize(img, (128, 128))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_flat = img_gray.flatten().reshape(1, -1)

    # Get prediction and probability
    prediction = model.predict(img_flat)[0]
    probabilities = model.predict_proba(img_flat)[0]
    confidence = round(probabilities[prediction] * 100, 2)  # Convert to percentage

    # Label mapping - matching exactly with training categories
    disease_labels = {
        0: "Healthy Leaf 🌿",
        1: "Leaf Spot Disease 🍂",
        2: "Sterility Mosaic Disease 🦠",
        3: "Leaf Webber Disease 🕸️"
    }

    # Debug print to check prediction
    print(f"Raw prediction number: {prediction}")
    
    # Map prediction to label
    result = disease_labels.get(prediction, "Unknown Disease")
    
    # Debug print to check mapping
    print(f"Mapped to disease: {result}")

    # Create a URL path for the saved image
    image_url = url_for('static', filename=f'uploads/{image.filename}')
    return render_template('result.html', 
                         result=result, 
                         image_url=image_url,
                         confidence=confidence)


if __name__ == '__main__':
    app.run(debug=True)
