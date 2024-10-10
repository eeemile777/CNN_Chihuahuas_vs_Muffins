import os
from flask import Flask, request, render_template, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from werkzeug.utils import secure_filename

# Define the Flask app
app = Flask(__name__)

# Load the pre-trained model
print("Loading model...")
model = load_model('optimal_model.keras', compile=False)
print("Model loaded successfully!")

# Allowed extensions for file upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Folder to save uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the image
def preprocess_image(img_path, img_size=64):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_normalized = img_resized / 255.0
    img_reshaped = np.reshape(img_normalized, (1, img_size, img_size, 1))
    return img_reshaped

# Function to classify the image
def classify_image(img_path):
    preprocessed_img = preprocess_image(img_path)
    prediction = model.predict(preprocessed_img)
    
    percentage = 1 - prediction[0][0]
    predicted_class = np.round(prediction[0][0])
    class_label = 'Chihuahua' if predicted_class == 0 else 'Muffin'
    rounded_percentage = round(percentage * 100, 2)
    
    return class_label, rounded_percentage

# Route to render the main page
@app.route('/')
def index():
    print("Rendering index page...")
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    print("Received image for prediction.")
    
    if 'Chihuhia-or-Muffin' not in request.files:
        print("No file part in the request.")
        return redirect(request.url)
    
    file = request.files['Chihuhia-or-Muffin']
    
    if file.filename == '':
        print("No file selected.")
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        label, confidence = classify_image(file_path)
        print(f"Prediction result: {label}, Confidence: {confidence}%")
        
        return jsonify({
            'label': label,
            'confidence': confidence
        })
    
    return redirect(request.url)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # Run the app with debugging enabled
    app.run(debug=True)
