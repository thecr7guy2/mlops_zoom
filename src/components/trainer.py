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
import wandb
import random
import string

class Trainer:
    def __init__(self,X_train,X_test,y_train,y_test,artifact_path):
        self.X_train = X_train
        self.X_test = X_test 
        self.y_train = y_train
        self.y_test = y_test
        # self.models = {"Random Forest": RandomForestRegressor(),
        #                "Decision Tree": DecisionTreeRegressor(),
        #                "Linear Regression" : LinearRegression()}
        self.models = {"Linear Regression" : LinearRegression(),
                       "Decision Tree": DecisionTreeRegressor()}
        self.model_path = artifact_path

    def _generate_random_string(self, length=6):
        """Generate a random string of fixed length."""
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))
        
    def evaluate_models(self):
        report ={}
        for i in range (len(self.models.keys())):
            model_n = list(self.models.keys())[i]
            random_str = self._generate_random_string()
            unique_run_name = f"{model_n}_{random_str}"
            with wandb.init(project="mlops", entity="thecr7guy3", name= unique_run_name):
                wandb.config.update({"model_name": list(self.models.keys())[i]})
                model = list(self.models.values())[i]
                print(f"Training {list(self.models.keys())[i]}")
                model.fit(self.X_train,self.y_train)
                print("Fitted the model on training Data")
                preds = model.predict(self.X_test)
                print("Generated predictions")
                model_score = r2_score(self.y_test,preds)
                rmse = mean_squared_error(self.y_test,preds,squared =False)
                mae = mean_absolute_error(self.y_test,preds)
                print("Calculated the metrics")
                report[list(self.models.keys())[i]] = model_score
                wandb.log({"R2 score": model_score})
                wandb.log({"Root Mean Squared Error": rmse})
                wandb.log({"Mean Absolute error": mae})
                print("Model evaluated")
        return report 
    
    def report_results(self):
        model_scores = self.evaluate_models()
        print("All Models evaluated")
        best_model_name = max(model_scores, key=model_scores.get)
        best_score = model_scores[best_model_name]
        best_model = self.models[best_model_name]
        self.save_best_model(best_model,best_model_name)
        print("Saved the best model")
        return best_model_name,best_score
    
    def save_best_model(self,model,model_name):
        model_file_name = f"{model_name.replace(' ', '_')}_best_model.joblib"
        dump(model, os.path.join(self.model_path, model_file_name))

