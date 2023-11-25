from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
import numpy as np

di_obj = DataIngestion("../../data/pre_proc.csv")
train_path,test_path = di_obj.execute()
dt_obj = DataTransformation("../../artifacts/",train_path,test_path)
a,b,c,d,e = dt_obj.execute()

print(a.shape)
print(b.shape)
print(c.shape)
print(d.shape)
print(e)

