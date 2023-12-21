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


# # Welcome to your prefect.yaml file! You can use this file for storing and managing
# # configuration for deploying your flows. We recommend committing this file to source
# # control along with your flow code.

# # Generic metadata about this project
# name: mlops_zoom
# prefect-version: 2.14.10

# # build section allows you to manage and build docker images
# build: null

# # push section allows you to manage if and how this project is uploaded to remote locations
# push: null

# # pull section allows you to provide instructions for cloning this project in remote locations
# pull:
# - prefect.deployments.steps.git_clone:
#     repository: https://github.com/thecr7guy2/mlops_zoom.git
#     branch: main
#     access_token: null
# - prefect.deployments.steps.pip_install_requirements:
#     directory: {{ clone-step.directory }}
#     requirements_file: requirements.txt
#     stream_output: False

# # the deployments section allows you to provide configuration for deploying flows
# deployments:
# - name: train_deployement
#   version: null
#   tags: []
#   description: null
#   schedule: {}
#   flow_name: null
#   entrypoint: train/pipelines/train_pipe.py:main_flow
#   parameters: {}
#   work_pool:
#     name: trypool_1
#     work_queue_name: default
#     job_variables: {}
