from train.components.data_ingestion import data_ingestion_flow
from train.components.data_transformation import data_transformation_flow
from train.components.trainer import training_flow
from prefect import flow,task

import mlflow
from mlflow import MlflowClient

import optuna
from optuna.integration.mlflow import MLflowCallback
import xgboost as xgb

import os

from prefect.blocks.system import Secret





@task(name="MLFlow Init")
def init_mlflow(mlflow_tracking_uri, mlflow_experiment_name):
    secret_block = Secret.load("accesskey")
    aws_access_key_id = secret_block.get()

    secret_block = Secret.load("secretkey")
    aws_secret_access_key = secret_block.get()

    os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key

    client = MlflowClient(mlflow_tracking_uri)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    try:
        experiment_id = mlflow.create_experiment(mlflow_experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name( mlflow_experiment_name).experiment_id

    try:
        experiment_id2 = mlflow.create_experiment("Hyper")
    except:
        experiment_id2 = mlflow.get_experiment_by_name("Hyper").experiment_id

    mlflow.set_experiment(experiment_id=experiment_id)
    mlflow.set_experiment(experiment_id=experiment_id2)

    optuna_mlflow_callback = MLflowCallback(tracking_uri=mlflow_tracking_uri,
                                            metric_name='rmse',
                                            create_experiment=False,
                                            )

    return client, optuna_mlflow_callback



@flow
def main_flow():
    mlflow_tracking_uri = "http://localhost:5000/"
    mlflow_experiment_name = "Car Price Prediction Best features"
    
    mlflow_client, optuna_mlflow_callback = init_mlflow(
        mlflow_tracking_uri, mlflow_experiment_name
    )
    # Data Ingestion
    train_path, test_path = data_ingestion_flow("data/pre_proc.csv")

   
    # Data Transformation
    X_train, X_test, y_train, y_test, transformer_path = data_transformation_flow("artifacts/", train_path, test_path)

   
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_test, label=y_test)

    best_params = training_flow(train,valid,optuna_mlflow_callback,mlflow_experiment_name,mlflow_client,train_path,test_path,transformer_path)


    print("sai")


if __name__ == "__main__":
    main_flow()