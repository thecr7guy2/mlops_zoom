import joblib
import pandas as pd

def load_model(model_path):
    """Load the pre-trained model from the specified path."""
    model = joblib.load(model_path)
    return model

def preprocess_input(input_data,transformer_path):
    """
    Preprocess the input data. 
    Adjust this function according to your model's requirements.
    """
    transformer = joblib.load(transformer_path)
    data_array = pd.DataFrame([input_data])
    processed_data =  transformer.transform(data_array) # Placeholder for actual preprocessing steps
    return processed_data

def make_prediction(model, data):
    """Make a prediction using the pre-trained model."""
    prediction = model.predict(data)
    return float(prediction[0])

def inference(input,model_path,transformer_path):
    model = load_model(model_path)
    processed_data = preprocess_input(input,transformer_path)
    prediction = make_prediction(model,processed_data)
    return prediction

