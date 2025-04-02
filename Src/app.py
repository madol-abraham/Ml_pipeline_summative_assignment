import tensorflow as tf
tf.config.run_functions_eagerly(True)  # Force eager execution
tf.data.experimental.enable_debug_mode()  # Additional debugging
from flask import Flask, request, jsonify, render_template, Response, send_from_directory
import numpy as np
import os
import time
import json
import logging
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from datetime import datetime
from retrain import ModelRetrainer
retrainer = ModelRetrainer()  # Initialize it
#result = retrainer.retrain("your_data_file.csv")
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config.update(
    SECRET_KEY=os.urandom(24),
    UPLOAD_FOLDER='uploads',
    ALLOWED_EXTENSIONS={'csv'},
    MODEL_PATH="models/cool_model.h5",
)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load or create model
try:
    model = load_model(app.config['MODEL_PATH'])
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise RuntimeError("Model loading failed")

@app.route("/")
def home():
    return render_template('home.html', active_page='home')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
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

            prediction = model.predict(input_features, verbose=0)
            predicted_prob = float(prediction[0][0])
            predicted_class = predicted_prob >= 0.5
            confidence = round(predicted_prob if predicted_class else 1 - predicted_prob, 4)

            return render_template('predict.html', 
                                active_page='predict',
                                show_result=True,
                                prediction=predicted_class,
                                confidence=confidence,
                                input_data=data)
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return render_template('predict.html', 
                                active_page='predict',
                                error="Prediction failed. Please check your inputs.")
    
    return render_template('predict.html', active_page='predict')
@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    if request.method == 'POST':
        if 'dataset' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
        
        file = request.files['dataset']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            epochs = int(request.form.get('epochs', 10))
            batch_size = int(request.form.get('batch_size', 32))
            
            return jsonify({
                'status': 'success',
                'filename': filename,  # Return the saved filename
                'epochs': epochs,
                'batch_size': batch_size
            })
    
    return render_template('retrain.html')
@app.route('/retrain/stream')
def retrain_stream():
    filename = request.args.get('filename')
    epochs = int(request.args.get('epochs', 10))
    batch_size = int(request.args.get('batch_size', 32))
    
    def generate():
        try:
            # Start retraining
            #result = ModelRetrainer.retrain(filename, epochs, batch_size)
            result = retrainer.retrain(filename, epochs, batch_size)
            
            if result['status'] == 'success':
                yield f"data: {json.dumps({
                    'progress': 100,
                    'message': result['message'],
                    'accuracy': result['accuracy'],
                    'status': 'completed'
                })}\n\n"
            else:
                yield f"data: {json.dumps({
                    'progress': 0,
                    'message': result['message'],
                    'status': 'error'
                })}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            yield f"data: {json.dumps({
                'progress': 0,
                'message': f"Stream error: {str(e)}",
                'status': 'error'
            })}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')
#root
@app.route('/Analytics')
def Analytics():
    # Check if images exist
    static_images = {
        'loss_curve': os.path.exists(os.path.join(app.static_folder, 'images/loss_curve.png')),
        'accuracy_curve': os.path.exists(os.path.join(app.static_folder, 'images/accuracy_curve.png')),
        'confusion_matrix': os.path.exists(os.path.join(app.static_folder, 'images/confusion_matrix.png'))
    }

    metrics = {
        'accuracy': 0.99,
        'precision': 0.97,
        'recall': 0.98,
        'f1_score': 0.98,
        'train_accuracy': 0.97,
        'val_accuracy': 0.99,
        'initial_loss': 0.91,
        'final_loss': 0.09,
        'loss_decreased': True,
        #last_trained': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'has_loss_curve': static_images['loss_curve'],
        'has_accuracy_curve': static_images['accuracy_curve'],
        'has_confusion_matrix': static_images['confusion_matrix']
    }
    
    return render_template('Analytics.html', 
                         active_page='Analytics',
                         metrics=metrics)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
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

        prediction = model.predict(input_features, verbose=0)
        predicted_prob = float(prediction[0][0])
        predicted_class = int(predicted_prob >= 0.5)
        confidence = round(predicted_prob if predicted_class else 1 - predicted_prob, 4)

        return jsonify({
            "irrigation_needed": predicted_class,
            "confidence": confidence,
            "raw_prediction": predicted_prob
        })
        
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({"error": "Prediction failed"}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)