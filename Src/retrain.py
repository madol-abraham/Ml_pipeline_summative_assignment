import os
import time
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure TensorFlow for optimal performance
tf.config.optimizer.set_jit(True)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.get_logger().setLevel('ERROR')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingLogger(Callback):
    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch: int, logs=None):
        elapsed = time.time() - self.start_time
        eta = (elapsed/(epoch+1)) * (self.total_epochs - epoch - 1)
        logger.info(
            f"Epoch {epoch+1}/{self.total_epochs} | "
            f"Loss: {logs['loss']:.4f} | "
            f"Acc: {logs['accuracy']:.4f} | "
            f"Val Loss: {logs['val_loss']:.4f} | "
            f"Val Acc: {logs['val_accuracy']:.4f} | "
            f"Elapsed: {elapsed:.1f}s | "
            f"ETA: {eta:.1f}s"
        )

class ModelRetrainer:
    def __init__(
        self,
        model_path: str = "models/cool_model.h5",
        upload_folder: str = "uploads",
        learning_rate: float = 0.001,
        dropout: float = 0.3,
        reg_strength: float = 0.01
    ):
        self.model_path = model_path
        self.upload_folder = upload_folder
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.reg_strength = reg_strength
        self.scaler = StandardScaler()
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(upload_folder, exist_ok=True)

    def _prepare_data(self, filename: str):
        filepath = os.path.join(self.upload_folder, filename)
        df = pd.read_csv(filepath)
        
        required_cols = [
            'Soil_Moisture', 'temperature', 'Time', 'Wind_speed_km_h',
            'Air_humidity_percent', 'rainfall', 'Soil_Type', 'Crop_Type',
            'Irrigation_Needed'
        ]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns in {filename}")
            
        X = df.drop('Irrigation_Needed', axis=1)
        y = df['Irrigation_Needed']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return (
            tf.convert_to_tensor(X_train, dtype=tf.float32),
            tf.convert_to_tensor(X_test, dtype=tf.float32),
            tf.convert_to_tensor(y_train, dtype=tf.float32),
            tf.convert_to_tensor(y_test, dtype=tf.float32)
        )

    def retrain(self, filename: str, epochs: int = 3, batch_size: int = 32):
        try:
            X_train, X_test, y_train, y_test = self._prepare_data(filename)
            
            if os.path.exists(self.model_path):
                model = load_model(self.model_path)
                logger.info("Loaded existing model")
            else:
                model = Sequential([
                    Dense(128, activation='relu', input_shape=(X_train.shape[1],),
                         kernel_regularizer=l2(self.reg_strength)),
                    Dropout(self.dropout),
                    Dense(64, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])
                logger.info("Created new model")
            
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[TrainingLogger(epochs)]
            )
            
            model.save(self.model_path)
            
            return {
                'status': 'success',
                'accuracy': float(history.history['val_accuracy'][-1]),
                'message': 'Training completed successfully'
            }
            
        except Exception as e:
            logger.error(f"Retraining failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }

retrainer = ModelRetrainer()