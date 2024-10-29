from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Set up the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your pre-trained model
model = load_model('optimal_model.keras')

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'fileInput' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['fileInput']
    
    # If the user does not select a file
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # If the file is allowed, save it and make predictions
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess the image for prediction
        image = Image.open(filepath).resize((224, 224))  # Adjust size as needed
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Predict using the model
        predictions = model.predict(image)
        confidence = round(float(np.max(predictions)) * 100, 2)
        label = 'Chihuahua' if np.argmax(predictions) == 0 else 'Muffin'
        
        return jsonify({'label': label, 'confidence': confidence})
    else:
        return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    # Create the upload folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)
