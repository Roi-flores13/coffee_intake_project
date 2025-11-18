from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


import pandas as pd
import numpy as np
from functools import partial
import optuna
from optuna.integration import MLflowCallback

import importlib
import sys
import os

import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv
from mlflow.models import infer_signature

from optuna_utils import define_search_space
from preprocessing import create_preprocessing_pipeline


load_dotenv(override=True) # Cargar las variables de entorno desde el archivo .env
EXPERIMENT_NAME = "/Users/roiflores.2213@gmail.com/coffee-intake-experiments" 

mlflow.set_tracking_uri("databricks")
experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)


df = pd.read_csv("../data/raw/synthetic_coffee_health_10000.csv")

X_train, X_test, y_train, y_test = train_test_split(df.drop("Sleep_Quality", axis=1), df["Sleep_Quality"],
                                                    test_size=0.2, random_state=88, stratify=df[["Sleep_Quality"]])

model_config = {
    "LogisticRegression": {"model_class":LogisticRegression(random_state=42)},
    "RandomForest": {"model_class": RandomForestClassifier(random_state=42)},
    "MLP": {"model_class": MLPClassifier(random_state=42)}
}
MAX_EVALS_PER_MODEL=15

columns_to_drop = [
    "Coffee_Intake",
    "Gender_Other",
    "Occupation_Other",
    "is_oceania",
    "Health_Issues",
    "Stress_Level",
]

def objective(trial, model_class, model_name):
    
    full_params = define_search_space(trial, model_name) #Uses the parameter space from optuna_utils.py
    
    model = model_class.set_params(**full_params) # Unpacks the parameters and sets up the model with those
    
    preprocessing_pipeline = create_preprocessing_pipeline(columns_to_drop) # Creates a pipeline using the preprocessing seen before
    
    full_trial_pipeline = Pipeline([
        ("preprocessor", preprocessing_pipeline), 
        ("model", model)
    ])
    
    score = cross_val_score(full_trial_pipeline, X_train, y_train,       # Cross validates the model with the parameter and returns the score
                            cv=3, scoring="f1_weighted", error_score="raise").mean() 
    
    return score


for model_name, config in model_config.items():
    with mlflow.start_run(run_name=f"{model_name}_HPO") as parent_run:
        
        kwargs = {"nested": True} #Indica que el run es nested
        
        mlflow_callback = MLflowCallback(
            tracking_uri="databricks", #Llama a databricks
            metric_name="f1_score", #Uses f1_score as metric
            create_experiment=False, # Doesn't create a new experiment
            mlflow_kwargs= kwargs)
        
        study = optuna.create_study(direction="maximize") # We want to maximize the f1_score
        
        obj_func = partial(
            objective, # Goes into objective function
            model_class=config["model_class"],
            model_name=model_name
        )
        
        study.optimize(
            obj_func,
            n_trials=MAX_EVALS_PER_MODEL,
            callbacks=[mlflow_callback]
        )
        
        best_f1_metric = study.best_value #Gets best f1_score
        best_params_cleaned = study.best_params #Gets best params
        
        print(f"--- 🏆 Mejor F1 para {model_name}: {best_f1_metric:.4f} ---")
        
        # --------------------------------------------------------------
        # This block of code sets up the fixed parameters for each model
        if model_name == 'LogisticRegression':
            best_params_cleaned['solver'] = 'saga'
            best_params_cleaned['max_iter'] = 1000
            
        elif model_name == 'MLP':
            best_params_cleaned['solver'] = 'adam'
            best_params_cleaned['max_iter'] = 500
        # --------------------------------------------------------------
        
        best_model_instance = config["model_class"].set_params(**best_params_cleaned) # Chooses the best model and its parameters
        
        # Creates a new pipeline to preprocess and fit with the whole data 
        final_preprocessing = create_preprocessing_pipeline(columns_to_drop) 
        
        final_production_pipeline = Pipeline([
            ("preprocessor", final_preprocessing),
            ("model", best_model_instance)
        ])
        final_production_pipeline.fit(X_train, y_train) # Re-trains the model, but now with all of the data
        
        example_data = X_train.iloc[:10] #Input data example
        example_predictions = final_production_pipeline.predict(example_data) # Output data example
        signature = infer_signature(example_data, example_predictions) #Creates signature for the model
        
        # 8. Loguea el pipeline final y la métrica en el Run Padre
        mlflow.log_params(best_params_cleaned) # Loguea los params limpios en el padre
        mlflow.log_metric("F1_score", best_f1_metric)
        
                
        mlflow.sklearn.log_model(
            sk_model=final_production_pipeline,
            artifact_path="model",
            code_paths=["preprocessing.py", "optuna_utils.py"],
            signature=signature,
            input_example=example_data
        )
        
model_registry = "workspace.default.coffee-intake-experiments"
runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME],
    order_by= ["metrics.F1_score DESC"],
    output_format="list"
)

if len(runs) > 0:
    best_run = runs[0]
    second_best = runs[1]
    
result_champ = mlflow.register_model(
    model_uri=f"runs:/{best_run.info.run_id}/model",
    name=model_registry
)

result_chall = mlflow.register_model(
    model_uri=f"runs:/{second_best.info.run_id}/model",
    name=model_registry
)

client = MlflowClient()
model_chall_version = result_chall.version # Callenger version
model_champ_version = result_champ.version # Champion version
challenger_alias ="Challenger"
champ_alias ="Champion"

# Challenger alias setter
client.set_registered_model_alias(
    name=model_registry,
    alias=challenger_alias,
    version=model_chall_version
)

# Champion alias setter
client.set_registered_model_alias(
    name=model_registry,
    alias=champ_alias,
    version= model_champ_version
)