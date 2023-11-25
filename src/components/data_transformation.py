import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import os
import numpy as np
from joblib import dump

class DataTransformation:
    def __init__(self,artifacts_path,train_df_path,test_df_path):
        self.artifacts_path = artifacts_path
        self.train_path = train_df_path
        self.test_path = test_df_path

    def read_data(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        return train_df,test_df

    def create_transformer_obj(self,df):
        numerical_cols = ['odometer', 'vehicle_age']
        categorical_cols = df.select_dtypes(include=['object']).columns


        numerical_pipeline = Pipeline([('scaler', StandardScaler())])
        categorical_pipeline = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
    
        preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)]
        )

        return preprocessor
    
    def execute(self):
        train_df,test_df= self.read_data()
        transformer_obj = self.create_transformer_obj(train_df)

        target_column_name="price"

        X_train_df=train_df.drop(columns=[target_column_name],axis=1)
        y_train_df=train_df[target_column_name]

        X_test_df=test_df.drop(columns=[target_column_name],axis=1)
        y_test_df=test_df[target_column_name]

        X_train=transformer_obj.fit_transform(X_train_df)
        y_train = np.array(y_train_df)

        X_test=transformer_obj.transform(X_test_df)
        y_test = np.array(y_test_df)




        # Now that I fit transfrom the column transformer I can use it whenever I later need it 
        a=os.path.join(self.artifacts_path,"transformer.joblib")
        dump(transformer_obj,a)

        return X_train, X_test, y_train, y_test,a 




