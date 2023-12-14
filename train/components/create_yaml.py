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
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski"],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
},
"DecisionTree": {
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 6],
    "max_features": [None, "auto", "sqrt", "log2"]
},
"GradientBoosting": {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 9],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 3, 5]
},
"AdaBoost": {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 1.0],
    "loss": ["linear", "square", "exponential"]
},
# "CatBoost": {
#     "iterations": [100, 300, 500],
#     "learning_rate": [0.01, 0.1, 0.2],
#     "depth": [4, 6, 8],
#     "l2_leaf_reg": [1, 3, 5],
#     "border_count": [32, 64, 128]
# }

}   

parser = argparse.ArgumentParser(description='Run a W&B sweep for a specified model.')
parser.add_argument('model_name', type=str, help='Name of the model for the sweep (e.g., RandomForest, KNN)')

args = parser.parse_args()
model_name = args.model_name

create_yaml = CreateYaml(model_name, model_hyperparameters)
create_yaml.create_sweep_config()