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
from evidently.metrics import DatasetDriftMetric

CREATE_TABLE_STATEMENT = """
CREATE TABLE IF NOT EXISTS drift_metrics (
    timestamp TIMESTAMP,
    number_of_columns INT,
    number_of_drifted_columns INT,
    dataset_drift BOOLEAN,
    drift_detected BOOLEAN
);
"""


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

    with psycopg2.connect("host=localhost port=5432 dbname=monitor user=postgres password=example") as conn:
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute(CREATE_TABLE_STATEMENT)
        cursor.close()

    with psycopg2.connect("host=localhost port=5432 dbname=user_data user=postgres password=example") as conn:
        conn.autocommit = True
        cursor = conn.cursor()
        count_query = "SELECT COUNT(*) FROM vehicle_data"
        cursor.execute(count_query)
        row_count = cursor.fetchone()[0]  
        cursor.close()

    return row_count

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


def get_ref_data(preprocessor):
    ref = pd.read_csv("../data/train_data.csv")
    ref = ref.drop(columns=['price'])
    temp_x_ref = preprocessor.transform(ref)
    return temp_x_ref,ref

def get_curr_data(preprocessor,ref):
    with psycopg2.connect("host=localhost port=5432 dbname=user_data user=postgres password=example") as conn:
        conn.autocommit = True
        cursor = conn.cursor()
        query = "SELECT * from vehicle_data"
        cursor.execute(query)
        tuples_list = cursor.fetchall()
        cursor.close()
        columns_pg = ref.columns.tolist()
        columns_pg.insert(0, 'id')
        curr = pd.DataFrame(tuples_list, columns=columns_pg)
        curr = curr.drop(columns="id")
        temp_x_curr = preprocessor.transform(curr)
        return temp_x_curr,curr
    # curr = pd.read_csv("../data/test_data.csv")
    # curr = curr.drop(columns=['price'])
    # temp_x_curr = preprocessor.transform(curr)
    # return temp_x_curr,curr



def get_predictions(model,data):
    return model.predict(data)

# def map_columns_evidently(numerical_features,categorical_features):
#     column_mapping = ColumnMapping()
#     column_mapping.numerical_features = numerical_features
#     column_mapping.categorical_features = categorical_features

def run_evidently(ref,curr):
    data_drift_dataset_report = Report(metrics=[DatasetDriftMetric()])
    data_drift_dataset_report.run(reference_data=ref,current_data=curr)
    result=data_drift_dataset_report.as_dict()
    return result

def send_metrics_to_db(result):

    num_drifted_columns = result['metrics'][0]['result']['number_of_drifted_columns']
    num_columns = result['metrics'][0]['result']['number_of_columns']
    dataset_drift = result['metrics'][0]['result']['dataset_drift']

    if num_drifted_columns/num_columns > 0.3 or dataset_drift:
        drift_detected = True
    else:
        drift_detected = False 
    

    with psycopg2.connect("host=localhost port=5432 dbname=monitor user=postgres password=example") as conn:
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute(
            "insert into drift_metrics(timestamp, number_of_columns, number_of_drifted_columns, dataset_drift, drift_detected) values (%s, %s, %s, %s, %s)",
            (
                datetime.now(),
                num_columns,
                num_drifted_columns,
                dataset_drift,
                drift_detected,
            ),
        )

    if drift_detected:
        #update the data (not possible for me :( )
        response = run_deployment(name='main-flow/train_deployement')
        print("Model Has now been retrained as Drift was detected")

    else:
        print("No drift detected, The model is still good to go. Metrics have been pushed to the db")
    


# @flow(name="monitor")
def monitor():
    ##############################################
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    ##############################################
    num_entries = prep_db()

    if num_entries<30:
        print("Moniotring not neeeded as insufficient data is present for monitoring")
    else:
        mlflow_client = init_mlflow(MLFLOW_TRACKING_URI)
        model = get_best_model(mlflow_client)
        pre_processor = get_transformer()
        pre_ref,ref = get_ref_data(pre_processor)
        pre_curr,curr = get_curr_data(pre_processor,ref)
        ref["prediction"] = get_predictions(model,pre_ref)
        curr["prediction"] = get_predictions(model,pre_curr)
        result = run_evidently(ref.head(1000),curr)
        print(result)
        send_metrics_to_db(result)

if __name__ == "__main__":
    monitor()




