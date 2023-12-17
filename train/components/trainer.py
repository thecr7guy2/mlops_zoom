from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
from joblib import dump
import os
from prefect import task,flow
import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType
import optuna
from optuna.integration.mlflow import MLflowCallback
import xgboost as xgb



@task(log_prints=True, name="Model hyperparameter tuning")
def hyperparameter_tuning(train,valid, optuna_mlflow_callback):
    
    def objective(trial):
        params = {
            'device': "cuda",
            'tree_method': 'hist',
            'objective':'reg:squarederror',
            'learning_rate': trial.suggest_categorical('learning_rate', [0.008, 0.01, 0.02, 0.05]),
            'random_state': 7,
            'max_depth': trial.suggest_int('max_depth', 3, 10), #change
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5), #change
            'lambda': trial.suggest_float('lambda', 1e-4, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),
        }
        
        booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=100,
                evals=[(valid, "validation")],
                early_stopping_rounds=10,
            )
        
        y_pred = booster.predict(valid)

        rmse = mean_squared_error(valid.get_label(), y_pred, squared=False)

        return rmse
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5, callbacks=[optuna_mlflow_callback])

    return study

@task(name="Best Experiment")
def find_best_run(study):
    car_price_prediction_experiment = dict(
        mlflow.get_experiment_by_name("Hyper")
    )

    experiment_id = car_price_prediction_experiment['experiment_id']

    best_run = mlflow.search_runs(
        experiment_ids=experiment_id,
        filter_string=f'metrics.rmse = {study.best_value}',
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
    )
    return best_run


@task(name="Get Best Run Params")
def get_best_params(best_run):
    if not best_run.empty:
        param_columns = [col for col in best_run.columns if col.startswith('params.')]
        best_params_df = best_run[param_columns]
        best_params_dict = best_params_df.iloc[0].to_dict()
        best_params_dict = {k.split('.', 1)[1]: float(v) for k, v in best_params_dict.items()}
        #best_params_dict["device"] = "cuda"
        best_params_dict["tree_method"] = "hist"
        best_params_dict["objective"] = "reg:squarederror"
        best_params_dict["random_state"] = 7
        best_params_dict["max_depth"]=int(best_params_dict["max_depth"])
        best_params_dict["min_child_weight"]=int(best_params_dict["min_child_weight"])
        return best_params_dict


@task(name="Train the best model")
def train_best_model(best_params,train,valid,train_path,test_path,trans_path):
    
    
    car_price_prediction_experiment = dict(mlflow.get_experiment_by_name("Car Price Prediction Best features"))

    experiment_id = car_price_prediction_experiment['experiment_id']

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(best_params)
        best_model = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=20,
        )

        y_pred = best_model.predict(valid)
        rmse = mean_squared_error(valid.get_label(), y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        mlflow.xgboost.log_model(best_model,artifact_path="models_mlflow")
        mlflow.log_artifact(train_path,'dataset')
        mlflow.log_artifact(test_path,'dataset')
        mlflow.log_artifact(trans_path)
    

@task(log_prints=True, name="Productionize the model")
def stage_model(client):

    car_price_prediction_experiment = dict(mlflow.get_experiment_by_name("Car Price Prediction Best features"))

    experiment_id = car_price_prediction_experiment['experiment_id']

    runs = client.search_runs(
        experiment_ids=experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=["metrics.rmse ASC"],
    )

    model_uri = f"runs:/{runs[0].info.run_id}/model"
    mlflow.register_model(
        model_uri=model_uri,
        name='used_car_prediction_best_xgboost'
    )

    if len(runs) > 1:
        model_uri = f"runs:/{runs[1].info.run_id}/model"
        mlflow.register_model(
            model_uri=model_uri,
            name='used_car_prediction_second_best_xgboost'
        )
    else:
        print("The second run does not exist. Model registration skipped.")
    
    
    model_version = client.search_model_versions("name='used_car_prediction_best_xgboost'")
    registered_model_version = max(model_version, key=lambda v: int(v.version)).version if model_version else None


    #model_version = client.search_model_versions("name='used_car_prediction_second_best_xgboost'")
    #contender_model_version = max(model_version, key=lambda v: int(v.version)).version if model_version else None
    
    client.set_registered_model_alias("used_car_prediction_best_xgboost", "champion", registered_model_version)
    #client.set_registered_model_alias("used_car_prediction_second_best_xgboost", "contender",contender_model_version)
    
    client.set_registered_model_tag("used_car_prediction_best_xgboost", "task", "regression")
   

@flow
def training_flow(train,valid,optuna_mlflow_callback,mlflow_experiment_name,mlflow_client,train_path,test_path,trans_path):
    study = hyperparameter_tuning(train,valid,optuna_mlflow_callback)
    best_run = find_best_run(study)
    best_params = get_best_params(best_run)
    train_best_model(best_params,train,valid,train_path,test_path,trans_path)
    stage_model(mlflow_client)
    return best_params
