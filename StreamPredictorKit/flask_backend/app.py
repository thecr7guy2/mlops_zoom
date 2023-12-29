import joblib
import pandas as pd
from flask import Flask,request,jsonify
import os 
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import psycopg2


mlflow_uri = os.getenv('MLFLOW_URL')
postgres_uri = os.getenv('POSTGRES_URL')

# mlflow_uri = "http://0.0.0.0:5000"
mlflow.set_tracking_uri(mlflow_uri)
client = MlflowClient(tracking_uri=mlflow_uri)

experiment_name = "Car Price Prediction Best features"
current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
experiment_id=current_experiment['experiment_id']
# print(experiment_id)

runs = client.search_runs(experiment_ids=[experiment_id], order_by=["start_time desc"], max_results=1)
runs = client.search_runs(experiment_ids=[experiment_id], order_by=["metrics.rmse ASC"], max_results=1)
#get the latest run 
latest_run_id = runs[0].info.run_id
best_run_id = runs[0].info.run_id

# print(latest_run_id)

mlflow.artifacts.download_artifacts("s3://pricemyride/1/"+latest_run_id+"/artifacts/transformer.joblib",dst_path="artifacts/",tracking_uri= mlflow_uri)
#get the latest transformer as the data may change 

transformer = joblib.load("artifacts/transformer.joblib") 
model_name = "used_car_prediction_best_xgboost"
alias = "champion"
model = mlflow.pyfunc.load_model(f"runs:/{best_run_id}/models_mlflow")
#getting the best model regardless of data


CREATE_TABLE_STATEMENT = ''' CREATE TABLE IF NOT EXISTS vehicle_data (
    id SERIAL PRIMARY KEY,
    region VARCHAR(255),
    price BIGINT,
    manufacturer VARCHAR(255),
    condition VARCHAR(255),
    cylinders VARCHAR(255),
    fuel VARCHAR(255),
    odometer DOUBLE PRECISION,
    transmission VARCHAR(255),
    drive VARCHAR(255),
    type VARCHAR(255),
    paint_color VARCHAR(255),
    vehicle_age DOUBLE PRECISION
);'''

conn = psycopg2.connect("host=database port=5432 user=postgres password=example")
conn.autocommit = True
cursor = conn.cursor()
query = cursor.execute("SELECT 1 FROM pg_database WHERE datname='user_data'")
res = cursor.fetchall()
if len(res) == 0:
    cursor.execute("create database user_data;")
    cursor.close()
    conn.close()

with psycopg2.connect("host=database port=5432 dbname=user_data user=postgres password=example") as conn:
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute(CREATE_TABLE_STATEMENT)
    cursor.close()

def insert_data(vehicle_data,price):
    int64_price = int(price)
    with psycopg2.connect("host=database port=5432 dbname=user_data user=postgres password=example") as conn:
        conn.autocommit = True
        cursor = conn.cursor()
        insert_query = """INSERT INTO vehicle_data (region, price, manufacturer, condition, cylinders, fuel, 
        odometer, transmission, drive, type, paint_color, vehicle_age)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""

        values = (
        vehicle_data["region"],
        int64_price,
        vehicle_data["manufacturer"],
        vehicle_data["condition"],
        vehicle_data["cylinders"],
        vehicle_data["fuel"],
        vehicle_data["odometer"],
        vehicle_data["transmission"],
        vehicle_data["drive"],
        vehicle_data["type"],
        vehicle_data["paint_color"],
        vehicle_data["vehicle_age"])

        cursor.execute(insert_query, values)
        cursor.close()
        



app = Flask("Used Car Price Prediction")

def inference(input_data,model,transformer):
    data_array = pd.DataFrame([input_data])
    processed_data =  transformer.transform(data_array)
    prediction = model.predict(processed_data)
    return float(prediction[0])

@app.route("/predict_car_price",methods=["POST"])
def predict_endpoint():
    input_data=request.get_json()
    vehicle_cost = inference(input_data,model,transformer)
    # print("@@@@@@@@@")
    # print(type(vehicle_cost))
    # print(vehicle_cost)
    
    return_dict = {"vehicle_cost":vehicle_cost}
    insert_data(input_data,vehicle_cost)
    return jsonify(return_dict)


if __name__== "__main__":
    app.run(debug=True,host="0.0.0.0",port=6969)