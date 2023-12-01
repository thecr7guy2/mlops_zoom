import yaml
import argparse

class CreateYaml():
    def __init__(self,model_name,model_hyperparameters):
        self.model_name = model_name
        for model, params in model_hyperparameters.items():
            for param, values in params.items():
                model_hyperparameters[model][param] = {"values": values}
        self.model_hyperparameters = model_hyperparameters
        
    def create_sweep_config(self):

        if self.model_name not in self.model_hyperparameters:
            raise ValueError(f"Hyperparameters for the model '{self.model_name}' are not defined.")
        
        parameters = {
            "model_type": {
                "value": self.model_name
            },
            **self.model_hyperparameters[self.model_name]
        }
        

        sweep_config = {
            "method": "grid",
            "metric": {
                "name": "r2_score",
                "goal": "maximize"
            },
            "program": "sweep.py",
            "name": self.model_name,
            "parameters": parameters
        }

        filename = "sweep_config.yaml"
        with open(filename, 'w') as file:
            yaml.dump(sweep_config, file, default_flow_style=False)

model_hyperparameters = {
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "max_features": [None,"sqrt", "log2"]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9],
        "metric": ["euclidean", "manhattan", "minkowski"]
    }
}   

parser = argparse.ArgumentParser(description='Run a W&B sweep for a specified model.')
parser.add_argument('model_name', type=str, help='Name of the model for the sweep (e.g., RandomForest, KNN)')

args = parser.parse_args()
model_name = args.model_name

create_yaml = CreateYaml(model_name, model_hyperparameters)
create_yaml.create_sweep_config()