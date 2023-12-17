import requests


vehicle_data = {
    "region": "auburn",
    "manufacturer": "ford",
    "condition": "excellent",
    "cylinders": "six",
    "fuel": "gas",
    "odometer": 1280.0,
    "transmission": "automatic",
    "drive": "rwd",
    "type": "truck",
    "paint_color": "black",
    "vehicle_age": 10.0
}

url = "http://54.159.125.199:6969/predict_car_price"

response = requests.post(url, json=vehicle_data)

print(response.json())


#  mlflow server \
    # --backend-store-uri sqlite:///mlflow.db \
    # --default-artifact-root s3://your-bucket-name/path/to/artifacts \
    # --host 0.0.0.0 --port 5000

#mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://pricemyride/ -h 0.0.0.0 -p 5000