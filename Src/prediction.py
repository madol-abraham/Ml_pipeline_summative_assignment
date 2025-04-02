import numpy as np
from tensorflow.keras.models import load_model
import logging

logger = logging.getLogger(__name__)

class IrrigationPredictor:
    def __init__(self, model_path="models/my_model.h5"):
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        try:
            model = load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError("Model loading failed")

    def predict(self, input_data):
        """
        Make irrigation prediction
        
        Args:
            input_data: dict containing:
                - Soil_Moisture: float
                - temperature: float
                - Time: float
                - Wind_speed_km_h: float
                - Air_humidity_percent: float
                - rainfall: float
                - Soil_Type: int
                - Crop_Type: int
                
        Returns:
            dict: {
                'irrigation_needed': int (0 or 1),
                'confidence': float,
                'raw_prediction': float
            }
        """
        try:
            input_features = np.array([[
                input_data['Soil_Moisture'],
                input_data['temperature'],
                input_data['Time'],
                input_data['Wind_speed_km_h'],
                input_data['Air_humidity_percent'],
                input_data['rainfall'],
                input_data['Soil_Type'],
                input_data['Crop_Type']
            ]], dtype=np.float32)

            prediction = self.model.predict(input_features, verbose=0)
            predicted_prob = float(prediction[0][0])
            predicted_class = int(predicted_prob >= 0.5)
            confidence = round(predicted_prob if predicted_class else 1 - predicted_prob, 4)

            return {
                'irrigation_needed': predicted_class,
                'confidence': confidence,
                'raw_prediction': predicted_prob
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise ValueError("Prediction failed")

# Singleton instance
predictor = IrrigationPredictor()