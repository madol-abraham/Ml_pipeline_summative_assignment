# Smartflow





![predict png](https://github.com/user-attachments/assets/85b8bb14-7f82-4a16-b702-6bdcc8095938)

![Screenshot (146)](https://github.com/user-attachments/assets/ed0207e0-e396-4e60-9807-ddb8e78aeb86)







# SmartFlow - Intelligent Irrigation Prediction System

![SmartFlow Logo](static/images/logo.png) <!-- Add your logo if available -->

## Project Description

SmartFlow is a machine learning-powered irrigation prediction system that helps farmers optimize water usage by predicting when crops need irrigation. The system analyzes environmental factors like soil moisture, temperature, humidity, and crop characteristics to make accurate predictions.

Key Features:
- Machine learning model for irrigation prediction
- Web interface for single predictions
- Bulk data upload capability
- Model retraining functionality
- Performance analytics dashboard
- Dockerized deployment

## Demo Video

[![SmartFlow Demo Video](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

*Replace YOUR_VIDEO_ID with your actual YouTube video ID*


## Setup Instructions

Follow these steps to set up the project locally:

### Prerequisites

- Python 3.8+
- Docker (optional)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/madol-abraham/Ml_pipeline_summative_assignment.git

cd Ml_pipeline_summative_assignment

### 2.Set up virtual environment
python -m venv venv

source venv/bin/activate  # On Windows use: venv\Scripts\activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Set up environment variables

Create a .env file in the root directory:
FLASK_APP=app.py

### 5. Run the application

The application will be available at http://localhost:5000

### Docker Setup

docker build -t Ml_pipeline_summative_assignment .
docker run -p 5000:5000 Ml_pipeline_summative_assignment


# File Structure

<pre> ``` Ml_pipeline_summative_assignment/
  │── README.md
  │── notebook/ 
  │   ├── ml_summative.ipynb 
  │── Src/ 
  
  │    ├── requirements.txt
  │    ├── Dockerfile 
  │    ├── app.py 
  │    ├──model.py 
  │    ├── prediction.py 
  │    ├──preprocessing.py 
  │    ├──retrain.py
  │── data/ 
  │   ├── train_data │
      ├── test_data 
      │── models/
  │       ├── cool_model.h5
  │── static/
  │    ├── css/ 
  │    │  ├── style.css 
  │    ├── images/ 
  │    ├── js/ 
  │    │  ├── upload.js
  │── templates/ 
  │   ├── base.html 
  │   ├── predict.html
  ] │ ├── retrain.html 
  │   ├── result.html
  │   ├── analytics.html 
  │   ├── home.html 
  │   ├── error.html ``` </pre>


    
   
