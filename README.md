# Smartflow





![predict png](https://github.com/user-attachments/assets/85b8bb14-7f82-4a16-b702-6bdcc8095938)

![Screenshot (146)](https://github.com/user-attachments/assets/ed0207e0-e396-4e60-9807-ddb8e78aeb86)







# SmartFlow - Intelligent Irrigation Prediction System


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

[Smartflow demo video](https://www.youtube.com/watch?v=aGR_Pc7LURc)


## Setup Instructions

Follow these steps to set up the project locally:

### Prerequisites

- Python 3.8+
- Docker (optional)
- Git

### 1. Clone the repository

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


# Usage
# Making Predictions
Navigate to the Predict page

Fill in the form with environmental and crop data

Submit to get irrigation prediction

# Retraining the Model
Navigate to the Retrain page

Upload a CSV file with a customized data

Click "Start Retraining"

Monitor progress on the dashboard

## API Endpoints    

POST /api/predict - you can access it here : http://172.17.0.2:5000/predict/

## Data Format
For retraining, upload CSV files with the following columns:

Soil_Moisture, temperature, Time,Wind_speed_km_h, Air_humidity_percent, rainfall, Soil_Type, Crop_Type, Irrigation_Needed

### Future improvements

# 1 Enhance model accuracy by :

adding more environmental data, testing advanced algorithms , and automating retraining with data drift detection.

# 2 Improve scalability and usability:

through cloud deployment , mobile app integration, and real-time GIS mapping for farmers.

# 3 Boost reliability and trust with model monitoring; 

explainable AI , and farmer feedback loops to refine predictions over time.
# 4 Integration of IoTs; 

To automatically trigger irrigation when the sensors detect that they is demand based on the environmental data in realtime.
## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## Author
Madol Abraham Kuol Madol

LinkdIn: [my linkdin](https://www.linkedin.com/in/madol-abraham-kuol-madol/)
Email: m.madol@alustudent.com

   
