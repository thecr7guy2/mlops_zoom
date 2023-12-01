from src.components.data_ingestion import data_ingestion_flow
from src.components.data_transformation import data_transformation_flow
from src.components.trainer import training_flow
from prefect import flow

@flow
def main_flow():
    # Data Ingestion
    train_path, test_path = data_ingestion_flow("data/pre_proc.csv")

    # Data Transformation
    X_train, X_test, y_train, y_test, transformer_path = data_transformation_flow("artifacts/", train_path, test_path)

    # Model Training and Evaluation
    print("Trainer object Initialised")
    best_model_name, best_score= training_flow(X_train, X_test, y_train, y_test, "artifacts/")
    print(best_model_name, best_score)

if __name__ == "__main__":
    main_flow()