import joblib
import pandas as pd
from flask import Flask,request,jsonify
import os 
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pyfunc

mlflow_uri = os.getenv('MLFLOW_URL')

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
    return_dict = {"vehicle_cost":vehicle_cost}
    return jsonify(return_dict)


if __name__== "__main__":
    app.run(debug=True,host="0.0.0.0",port=6969)