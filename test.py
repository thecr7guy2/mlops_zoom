import requests


vehicle_data = {
    "region": "auburn",
    "manufacturer": "ford",
    "condition": "excellent",
    "cylinders": "six",
    "fuel": "gas",
    "odometer": 128000.0,
    "transmission": "automatic",
    "drive": "rwd",
    "type": "truck",
    "paint_color": "black",
    "vehicle_age": 10.0
}

url = "http://127.0.0.1:6969/predict_car_price"

response = requests.post(url, json=vehicle_data)

print(response.json())