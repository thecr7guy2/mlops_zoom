import pandas as pd
from sklearn.model_selection import train_test_split
import os
from prefect import task,flow
import mlflow

@task(retries=5, retry_delay_seconds=5)
def read_data(csv_path):
    """Read data from CSV file."""
    try:
        data = pd.read_csv(csv_path)
        return data
    except Exception as e:
        raise Exception(f"Error reading the CSV file: {e}")

@task(retries=5, retry_delay_seconds=5)
def basic_preprocessing(data):
    """Preprocess the data."""
    # Add your data preprocessing steps here
    data = data.dropna()
    #Other basic preprocessing stuff 
    return data

@task(retries=5, retry_delay_seconds=5)
def split_and_save_data(data, csv_path, test_size, random_state):
    """Split the data into train and test sets and save to CSV files."""
    X_train, X_test = train_test_split(data, test_size=test_size, random_state=random_state)

    # Define file paths
    base_path = os.path.dirname(csv_path)
    train_file_path = os.path.join(base_path, 'train_data.csv')
    test_file_path = os.path.join(base_path, 'test_data.csv')

    # Save files
    X_train.to_csv(train_file_path, index=False)
    X_test.to_csv(test_file_path, index=False)

   
    return train_file_path, test_file_path

@flow
def data_ingestion_flow(csv_path, test_size=0.2, random_state=42, preprocess=False):
    """Execute the data ingestion process."""
    data = read_data(csv_path)
    if preprocess:
        data = basic_preprocessing(data)
    else:
        pass
    train_path, test_path = split_and_save_data(data, csv_path, test_size, random_state)
    
    return train_path, test_path
