import wandb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from joblib import dump
from data_ingestion import DataIngestion
from data_transformation import DataTransformation
import os

def get_model(config):
    if config.model_type == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=config.get("n_estimators", 100),
            max_features=config.get("max_features", None),
            max_depth=config.get("max_depth", None)
        )
    elif config.model_type == "KNN":
        model = KNeighborsRegressor(
            n_neighbors=config.get("n_neighbors", 5),
            weights=config.get("weights", "uniform"),
            metric=config.get("metric", "minkowski"),
            algorithm=config.get("algorithm", "auto")
        )
    elif config.model_type == "DecisionTree":
        model = DecisionTreeRegressor(
            max_depth=config.get("max_depth", None),
            min_samples_split=config.get("min_samples_split", 2),
            min_samples_leaf=config.get("min_samples_leaf", 1),
            max_features=config.get("max_features", None)
        )
    elif config.model_type == "GradientBoosting":
        model = GradientBoostingRegressor(
            n_estimators=config.get("n_estimators", 100),
            learning_rate=config.get("learning_rate", 0.1),
            max_depth=config.get("max_depth", 3),
            min_samples_split=config.get("min_samples_split", 2),
            min_samples_leaf=config.get("min_samples_leaf", 1)
        )
    elif config.model_type == "AdaBoost":
        model = AdaBoostRegressor(
            n_estimators=config.get("n_estimators", 50),
            learning_rate=config.get("learning_rate", 1.0),
            loss=config.get("loss", "linear")
        )
    # elif config.model_type == "CatBoost":
    #     model = CatBoostRegressor(
    #         iterations=config.get("iterations", 100),
    #         learning_rate=config.get("learning_rate", 0.1),
    #         depth=config.get("depth", 6),
    #         l2_leaf_reg=config.get("l2_leaf_reg", 3),
    #         border_count=config.get("border_count", 32)
    #     )
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

def save_and_log_model(model,config,model_path):
    # Save model to file
    model_name = config.model_type 
    model_path = os.path.join(model_path,f"{model_name}.joblib")
    dump(model, model_path)

    # Create a wandb artifact for the model
    artifact = wandb.Artifact(name=model_name, type='model', description=f"Trained {model_name} model")
    artifact.add_file(model_path)

    # Log the model artifact
    wandb.log_artifact(artifact)

def main():
    with wandb.init(project="mlops", entity="thecr7guy3") as run:
        config = run.config
        di_obj = DataIngestion("../../data/pre_proc.csv")
        train_path,test_path = di_obj.execute()
        dt_obj = DataTransformation("../../artifacts/",train_path,test_path)
        X_train, X_test, y_train, y_test,a= dt_obj.execute()
        model = get_model(config)
        metrics = train_and_evaluate(model,X_train, X_test, y_train, y_test)
        #save_and_log_model(model,config,"../../artifacts/")
        wandb.log(metrics)

if __name__ == "__main__":
    main()







