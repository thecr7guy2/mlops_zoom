#!/usr/bin/env bash

# Start the Prefect server
echo "Starting Prefect server..."
pipenv run prefect server start &

sleep 3

pipenv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api

# Wait for the Prefect server to start
sleep 10

# Start the MLflow server
echo "Starting MLflow server..."
pipenv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://pricemyride/ -h 0.0.0.0 -p 5000 &

# Wait for the MLflow server to start
sleep 10

echo "Servers are running."
