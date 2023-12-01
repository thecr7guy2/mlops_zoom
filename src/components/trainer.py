# from src.components.data_ingestion import DataIngestion
# from src.components.data_transformation import DataTransformation
# import numpy as np

# di_obj = DataIngestion("../../data/pre_proc.csv")
# train_path,test_path = di_obj.execute()
# dt_obj = DataTransformation("../../artifacts/",train_path,test_path)
# a,b,c,d,e = dt_obj.execute()

# print(a.shape)
# print(b.shape)
# print(c.shape)
# print(d.shape)
# print(e)

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
from joblib import dump
import os
from prefect import task,flow

@task(retries=5, retry_delay_seconds=5, log_prints=True)
def evaluate_models(X_train, y_train, X_test, y_test, models):
    report = {}
    for model_name, model in models.items():
        print(f"Training {model_name}")
        model.fit(X_train, y_train)
        print("Fitted the model on training Data")
        preds = model.predict(X_test)
        print("Generated predictions")
        model_score = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        print("Calculated the metrics")
        report[model_name] = model_score
        print("Model evaluated")
    return report, models

@task(retries=5, retry_delay_seconds=5)
def report_results(model_scores, models, artifact_path):
    print("All Models evaluated")
    best_model_name = max(model_scores, key=model_scores.get)
    best_score = model_scores[best_model_name]
    best_model = models[best_model_name]
    save_best_model(best_model, best_model_name, artifact_path)
    print("Saved the best model")
    return best_model_name, best_score

def save_best_model(model, model_name, artifact_path):
    model_file_name = f"{model_name.replace(' ', '_')}_best_model.joblib"
    dump(model, os.path.join(artifact_path, model_file_name))

@flow
def training_flow(X_train, X_test, y_train, y_test, artifact_path):
    models = {"Linear Regression": LinearRegression(),
              "Decision Tree": DecisionTreeRegressor()}
    model_scores, models = evaluate_models(X_train, y_train, X_test, y_test, models)
    best_model_name, best_score= report_results(model_scores, models, artifact_path)
    return best_model_name, best_score