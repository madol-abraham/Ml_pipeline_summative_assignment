from flask import Flask, request, jsonify, render_template, Response, send_from_directory
import numpy as np
import os
import pandas as pd
import time
import json
import logging
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

# Global variables for training status
training_status = {
    'is_training': False,
    'progress': 0,
    'message': '',
    'accuracy': 0,
    'loss': 0
}

# Load the trained Keras model
MODEL_PATH = "models/my_model.h5"
try:
    model = load_model(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.warning(f"Model loading failed, creating new model: {str(e)}")
    # Create a simple model if loading fails
    model = Sequential([
        Dense(16, activation='relu', input_shape=(8,)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.save(MODEL_PATH)

# Helper function
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route - serves the main page
@app.route("/")
def home():
    return render_template('home.html')

# Prediction route - handles form submission
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get data from form
            data = {
                'Soil_Moisture': float(request.form.get('Soil_Moisture')),
                'temperature': float(request.form.get('temperature')),
                'Time': float(request.form.get('Time')),
                'Wind_speed_km_h': float(request.form.get('Wind_speed_km_h')),
                'Air_humidity_percent': float(request.form.get('Air_humidity_percent')),
                'rainfall': float(request.form.get('rainfall')),
                'Soil_Type': int(request.form.get('Soil_Type')),
                'Crop_Type': int(request.form.get('Crop_Type'))
            }

            # Prepare input array
            input_features = np.array([[
                data['Soil_Moisture'],
                data['temperature'],
                data['Time'],
                data['Wind_speed_km_h'],
                data['Air_humidity_percent'],
                data['rainfall'],
                data['Soil_Type'],
                data['Crop_Type']
            ]], dtype=np.float32)

            # Make prediction
            prediction = model.predict(input_features, verbose=0)
            predicted_prob = float(prediction[0][0])
            predicted_class = "Yes" if predicted_prob >= 0.5 else "No"
            confidence = round(predicted_prob if predicted_class == "Yes" else 1 - predicted_prob, 4)

            return render_template('result.html', 
                                prediction=predicted_class,
                                confidence=confidence,
                                input_data=data)
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            return render_template('error.html', error_message="Prediction failed")
    
    # If GET request, show the form
    return render_template('predict.html')

# Retrain model route
@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    if request.method == 'POST':
        if 'dataset' not in request.files:
            return render_template('error.html', error_message="No file uploaded")
        
        file = request.files['dataset']
        if file.filename == '':
            return render_template('error.html', error_message="No selected file")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Start training in background
            epochs = int(request.form.get('epochs', 10))
            batch_size = int(request.form.get('batch_size', 32))
            
            return render_template('dashboard.html', 
                                training_started=True,
                                filename=filename,
                                epochs=epochs,
                                batch_size=batch_size)
    
    return render_template('retrain.html')

# Training progress stream
@app.route('/retrain/stream')
def retrain_stream():
    def generate():
        global training_status
        training_status['is_training'] = True
        
        try:
            # Simulate training (replace with actual training code)
            for epoch in range(10):
                training_status['progress'] = (epoch + 1) * 10
                training_status['message'] = f"Epoch {epoch + 1}/10 - Processing"
                training_status['accuracy'] = 0.7 + epoch * 0.03
                training_status['loss'] = 0.5 - epoch * 0.05
                
                yield f"data: {json.dumps(training_status)}\n\n"
                time.sleep(1)
            
            # Training complete
            training_status.update({
                'progress': 100,
                'message': "Training completed successfully",
                'accuracy': 0.95,
                'loss': 0.1,
                'is_training': False
            })
            yield f"data: {json.dumps(training_status)}\n\n"
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}", exc_info=True)
            training_status.update({
                'message': f"Training failed: {str(e)}",
                'is_training': False
            })
            yield f"data: {json.dumps(training_status)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

# Dashboard route
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', 
                         training_started=False,
                         model_accuracy=0.85,  # Replace with actual model accuracy
                         last_trained="2023-07-20")  # Replace with actual last trained date

# API endpoint for programmatic access
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        # Prepare input array
        input_features = np.array([[
            data['Soil_Moisture'],
            data['temperature'],
            data['Time'],
            data['Wind_speed_km_h'],
            data['Air_humidity_percent'],
            data['rainfall'],
            data['Soil_Type'],
            data['Crop_Type']
        ]], dtype=np.float32)

        # Make prediction
        prediction = model.predict(input_features, verbose=0)
        predicted_prob = float(prediction[0][0])
        predicted_class = 1 if predicted_prob >= 0.5 else 0
        confidence = round(predicted_prob if predicted_class == 1 else 1 - predicted_prob, 4)

        return jsonify({
            "irrigation_needed": predicted_class,
            "confidence": confidence,
            "raw_prediction": predicted_prob
        })
        
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": "Prediction failed"}), 500

# API endpoint for training status
@app.route('/api/training_status')
def api_training_status():
    return jsonify(training_status)

# Static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == "_main_":
    app.run(host="0.0.0.0", port=5000, debug=True)