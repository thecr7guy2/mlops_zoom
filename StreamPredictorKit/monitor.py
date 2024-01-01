from prefect import task, flow
from prefect.deployments import run_deployment
import random
import mlflow
from mlflow import MlflowClient
from prefect.blocks.system import Secret
import os 


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

def get_best_model(client):
    experiment_name = "Car Price Prediction Best features"
    current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
    experiment_id=current_experiment['experiment_id']
    best_run = client.search_runs(experiment_ids=[experiment_id], order_by=["metrics.rmse ASC"], max_results=1)
    best_run_id = best_run[0].info.run_id
    model = mlflow.pyfunc.load_model(f"runs:/{best_run_id}/models_mlflow")
    return model

def get_transformer():
    


    


# @flow(name="monitor")
def monitor():
    status()
    
if __name__ == "__main__":
    monitor()




