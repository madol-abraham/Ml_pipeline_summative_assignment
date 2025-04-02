import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "soil_moisture": 25.5,
    "temperature": 22.0,
    "time": 12,
    "wind_speed": 5.0,
    "air_humidity": 80.0,
    "rainfall": 15.0,
    "soil_type": 1,
    "crop_type": 2
}

response = requests.post(url, json=data)
print(response.json())
