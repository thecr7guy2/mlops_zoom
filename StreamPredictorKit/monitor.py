from prefect import task, flow
from prefect.deployments import run_deployment
import random
import mlflow
from mlflow import MlflowClient
from prefect.blocks.system import Secret
import os 
import joblib
import pandas as pd
from datetime import datetime

import psycopg2
from evidently.report import Report
from evidently.metrics import DatasetCorrelationsMetric


# @task(name="Initilaize Mlflow and set aws environment")
def init_mlflow(mlflow_tracking_uri):
    secret_block = Secret.load("accesskey")
    aws_access_key_id = secret_block.get()

    secret_block = Secret.load("secretkey")
    aws_secret_access_key = secret_block.get()

    os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key

    client = MlflowClient(mlflow_tracking_uri)
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    return client

def prep_db():
    conn = psycopg2.connect("host=localhost port=5432 user=postgres password=example")
    conn.autocommit = True
    cursor = conn.cursor()
    query = cursor.execute("SELECT 1 FROM pg_database WHERE datname='monitor'")
    res = cursor.fetchall()
    if len(res) == 0:
        cursor.execute("create database monitor;")
        cursor.close()
        conn.close()
    else:
        cursor.close()
        conn.close()

def get_best_model(client):
    experiment_name = "Car Price Prediction Best features"
    current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
    experiment_id=current_experiment['experiment_id']
    best_run = client.search_runs(experiment_ids=[experiment_id], order_by=["metrics.rmse ASC"], max_results=1)
    best_run_id = best_run[0].info.run_id
    model = mlflow.pyfunc.load_model(f"runs:/{best_run_id}/models_mlflow")
    return model

def get_transformer():
    preprocessor= joblib.load("../artifacts/transformer.joblib")
    return preprocessor

def set_variables():
    prediction = 'prediction'
    numerical_features = ["odometer","vehicle_age"]
    categorical_features = ['region', 'manufacturer', 'condition', 'cylinders',
                             'fuel', 'transmission', 'drive', 'type', 'paint_color']
    
    return prediction,numerical_features,categorical_features

def get_ref_data(preprocessor,numerical_features,categorical_features):
    ref = pd.read_csv("../data/train_data.csv")
    temp_x_ref = preprocessor.transform(ref[numerical_features + categorical_features])
    return temp_x_ref,ref

def get_curr_data(preprocessor,numerical_features,categorical_features):
    curr = pd.read_csv("../data/test_data.csv")
    temp_x_curr = preprocessor.transform(curr[numerical_features + categorical_features])
    return temp_x_curr,curr

def get_predictions(model,data):
    return model.predict(data)

# def map_columns_evidently(numerical_features,categorical_features):
#     column_mapping = ColumnMapping()
#     column_mapping.numerical_features = numerical_features
#     column_mapping.categorical_features = categorical_features

def run_evidently(ref,curr):
    data_quality_dataset_report = Report(metrics=[DatasetCorrelationsMetric()])
    data_quality_dataset_report.run(reference_data=ref, current_data=curr)
    result = data_quality_dataset_report.as_dict()
    return result

def send_metrics_to_db(result):

    total_drift_detected = 2828

    with psycopg2.connect("host=localhost port=5432 dbname=monitor user=postgres password=example") as conn:
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute(
            "insert into evidently_drift_metrics(timestamp, model_drift, mmd_drift,cosine_dist_drift, total_drift_detected) values (%s, %s, %s, %s, %s)",
            (
                datetime.now(),
                model_drift,
                mmd_drift,
                cosine_dist_drift,
                total_drift_detected,
            ),
        )

    if total_drift_detected:
        response = run_deployment(name='main-flow/train_deployement')
    


# @flow(name="monitor")
def monitor():
    ##############################################
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    ##############################################
    prep_db()
    mlflow_client = init_mlflow(MLFLOW_TRACKING_URI)
    model = get_best_model(mlflow_client)
    pre_processor = get_transformer()
    prediction,numerical_features,categorical_features=set_variables()
    pre_ref,ref = get_ref_data(pre_processor,numerical_features,categorical_features)
    pre_curr,curr = get_curr_data(pre_processor,numerical_features,categorical_features)
    ref["prediction"] = get_predictions(model,pre_ref)
    curr["prediction"] = get_predictions(model,pre_curr)
    result = run_evidently(ref,curr)


    
if __name__ == "__main__":
    monitor()




