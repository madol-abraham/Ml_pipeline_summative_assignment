# Smartflow





![predict png](https://github.com/user-attachments/assets/85b8bb14-7f82-4a16-b702-6bdcc8095938)

![Screenshot (146)](https://github.com/user-attachments/assets/ed0207e0-e396-4e60-9807-ddb8e78aeb86)







# project overview

smartFlow is a machine learning classification model that utilizes a neural network to predict whether irrigation is needed based on various environmental factors.
The model is designed to assist farmers in optimizing water usage, improving crop yield, and conserving resources.It is an end-to-end solution which features retraining, prediction and visualisation.

# Features

# Binary Classification:  
Predicts whether irrigation is required (1) or not (0).

# Neural Network Model: 
Uses deep learning for accurate predictions.

# Optimized for Smart Agriculture: 
Specifically design to  automate irrigation decisions based on real-time inputs.

# Inputs and Features
The model takes the following environmental and soil conditions as input:

# Soil moisture
# Time
# Air Humidity
# Wind Speed (Km/h)
# Soil Type
# Temperature
# Crop Type
# Rainfall

# File Structure

<pre> ```plaintext Ml_pipeline_summative_assignment/ │── README.md │── requirements.txt │── Dockerfile │── app.py │── model.py │── prediction.py │── preprocessing.py │── retrain.py │── data/ │ ├── train_data.csv │ ├── test_data.csv │── models/ │ ├── cool_model.h5 │── notebook/ │ ├── ml_summative.ipynb │── static/ │ ├── css/ │ │ ├── style.css │ ├── images/ │ ├── js/ │ │ ├── upload.js │── templates/ │ ├── base.html │ ├── predict.html │ ├── retrain.html │ ├── result.html │ ├── analytics.html │ ├── home.html │ ├── error.html ``` </pre>
