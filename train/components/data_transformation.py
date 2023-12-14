import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import os
import numpy as np
from joblib import dump
from prefect import task,flow

@task(retries=5, retry_delay_seconds=5)
def read_train_test(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

@task(retries=5, retry_delay_seconds=5)
def create_transformer_obj(train_df):
    numerical_cols = ['odometer', 'vehicle_age']
    categorical_cols = train_df.select_dtypes(include=['object']).columns

    numerical_pipeline = Pipeline([('scaler', StandardScaler())])
    categorical_pipeline = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])
    return preprocessor

@task(retries=5, retry_delay_seconds=5)
def apply_transformer_obj(train_df, test_df, preprocessor):
    target_column_name = "price"

    X_train_df = train_df.drop(columns=[target_column_name], axis=1)
    y_train_df = train_df[target_column_name]

    X_test_df = test_df.drop(columns=[target_column_name], axis=1)
    y_test_df = test_df[target_column_name]

    X_train = preprocessor.fit_transform(X_train_df)
    y_train = np.array(y_train_df)

    X_test = preprocessor.transform(X_test_df)
    y_test = np.array(y_test_df)

    return X_train, X_test, y_train, y_test

@flow
def data_transformation_flow(artifacts_path, train_df_path, test_df_path):
    train_df, test_df = read_train_test(train_df_path, test_df_path)
    
    preprocessor = create_transformer_obj(train_df)
    X_train, X_test, y_train, y_test = apply_transformer_obj(train_df, test_df, preprocessor)
    transformer_path = os.path.join(artifacts_path, "transformer.joblib")
    dump(preprocessor, transformer_path)
    return X_train, X_test, y_train, y_test, transformer_path


 
        



