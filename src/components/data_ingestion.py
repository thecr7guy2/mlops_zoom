import pandas as pd
from sklearn.model_selection import train_test_split
import os

class DataIngestion:
    def __init__(self, csv_file_path, test_size=0.2, random_state=42,pre_process=False) :
        self.csv_path = csv_file_path
        self.test_size = test_size
        self.random_state = random_state
        self.preprocess = pre_process
    
    def read_data(self):
        """Read data from CSV file."""
        try:
            data = pd.read_csv(self.csv_path)
            return data
        except Exception as e:
            raise Exception(f"Error reading the CSV file: {e}")
        
    def preprocess_data(self, data):
        """Preprocess the data."""
        # Add your data preprocessing steps here
        # Example: data = data.dropna()
        # drop_col = ['url', 'region_url', 'title_status', 'VIN', 'size', 'image_url', 'description', 'lat','long', 'id','county']
        # data = data.drop(columns=drop_col)
        data = data.dropna()
        return data

    def split_and_save_data(self, data):
        """Split the data into train and test sets and save to CSV files."""
        X_train, X_test = train_test_split(data, test_size=self.test_size, random_state=self.random_state)

        # Define file paths
        base_path = os.path.dirname(self.csv_path)
        self.train_file_path = os.path.join(base_path, 'train_data.csv')
        self.test_file_path = os.path.join(base_path, 'test_data.csv')

        # Save files
        X_train.to_csv(self.train_file_path, index=False)
        X_test.to_csv(self.test_file_path, index=False)


    def execute(self):
        """Execute the data ingestion process."""
        data = self.read_data()
        if self.preprocess == True:
            processed_data = self.preprocess_data(data)
        else:
            pass
        self.split_and_save_data(data)
        return self.train_file_path, self.test_file_path

