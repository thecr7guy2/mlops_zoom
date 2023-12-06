import joblib
import pandas as pd
from flask import Flask,request,jsonify


model_path = "artifacts/KNN_Regression_best_model.joblib"
transformer_path = "artifacts/transformer.joblib"

app = Flask("Used Car Price Prediction")

def inference(input_data,model_path,transformer_path):
    model = joblib.load(model_path)
    transformer = joblib.load(transformer_path)
    data_array = pd.DataFrame([input_data])
    processed_data =  transformer.transform(data_array)
    prediction = model.predict(processed_data)
    return float(prediction[0])

@app.route("/predict_car_price",methods=["POST"])
def predict_endpoint():
    input_data=request.get_json()
    vehicle_cost = inference(input_data,model_path,transformer_path)
    return_dict = {"vehicle_cost":vehicle_cost}
    return jsonify(return_dict)


if __name__== "__main__":
    app.run(debug=True,host="0.0.0.0",port=6969)