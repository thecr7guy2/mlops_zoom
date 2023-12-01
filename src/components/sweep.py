import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from joblib import dump
from data_ingestion import DataIngestion
from data_transformation import DataTransformation

def get_model(config):
    if config.model_type == "RandomForest":
        model = RandomForestRegressor(
                n_estimators=config.get("n_estimators", 100),
                max_features = config.get("max_features",None),
                max_depth=config.get("max_depth", None)
            )
    elif config.model_type == "KNN":
        model = KNeighborsRegressor(
                n_neighbors=config.get("n_neighbors", 5),
                metric = config.get("metric",'minkowski')
            )
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")
    return model
   
        
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "r2_score": r2_score(y_test, preds),
        "rmse": mean_squared_error(y_test, preds, squared=False),
        "mae": mean_absolute_error(y_test, preds)
    }

def main():
    with wandb.init(project="mlops", entity="thecr7guy3") as run:
        config = run.config
        di_obj = DataIngestion("../../data/pre_proc.csv")
        train_path,test_path = di_obj.execute()
        dt_obj = DataTransformation("../../artifacts/",train_path,test_path)
        X_train, X_test, y_train, y_test,a= dt_obj.execute()
        model = get_model(config)
        metrics = train_and_evaluate(model,X_train, X_test, y_train, y_test)
        wandb.log(metrics)

if __name__ == "__main__":
    main()







