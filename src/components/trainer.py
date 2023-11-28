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
from sklearn.metrics import r2_score
from joblib import dump
import os 

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
                       "Decision Tree": DecisionTreeRegressor(),
                       "KNN Regression": KNeighborsRegressor()}
        self.metric = r2_score
        self.model_path = artifact_path
        
    def evaluate_models(self):
        report ={}
        for i in range (len(self.models.keys())):
            model = list(self.models.values())[i]
            print("Sai1")
            model.fit(self.X_train,self.y_train)
            print("Sai2")
            preds = model.predict(self.X_test)
            print("Sai3")
            model_score = self.metric(self.y_test,preds)
            print("Sai4")
            report[list(self.models.keys())[i]] = model_score
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

