import mlflow
from mlflow import MlflowClient

import


@task(name="MLFlow Init")
def init_mlflow(mlflow_tracking_uri):
    client = MlflowClient(mlflow_tracking_uri)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    return client


