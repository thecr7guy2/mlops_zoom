from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
import numpy as np
from src.components.trainer import Trainer

di_obj = DataIngestion("data/pre_proc.csv")
train_path,test_path = di_obj.execute()
dt_obj = DataTransformation("artifacts/",train_path,test_path)
X_train, X_test, y_train, y_test,a= dt_obj.execute()
tr_obj = Trainer(X_train,X_test,y_train,y_test,"artifacts/")
print("Trainer object Initialised")
print("Intializing wandb sweep to choose the model with best params")
c,d = tr_obj.report_results()
print(c,d)
