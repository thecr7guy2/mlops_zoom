PREFECT_PORT=4200
MLFLOW_PORT=5000

# Stop the Prefect server
echo "Stopping Prefect server..."
fuser -k ${PREFECT_PORT}/tcp

# Wait for the Prefect server to stop
sleep 5

# Stop the MLflow server
echo "Stopping MLflow server..."
fuser -k ${MLFLOW_PORT}/tcp

# Wait for the MLflow server to stop
sleep 5

echo "Servers have been stopped."
