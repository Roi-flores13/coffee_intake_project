from fastapi import FastAPI, HTTPException
import os 
from dotenv import load_dotenv
import mlflow 
from mlflow import MlflowClient
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

load_dotenv(override=True)

mlflow.set_tracking_uri("databricks")
client = MlflowClient()

EXPERIMENT_NAME = "/Users/roiflores.2213@gmail.com/coffee-intake-experiments"

model_name = "workspace.default.coffee-intake-experiments"
model_alias = "champion"
model_uri = f"models:/{model_name}@{model_alias}"
champion_model = mlflow.pyfunc.load_model(model_uri=model_uri)

class InputData(BaseModel):
    ID: int
    Age: int
    Gender: str
    Country: str
    Coffee_Intake: float
    Caffeine_mg: float
    Sleep_Hours: float
    BMI: float
    Heart_Rate: int
    Stress_Level: str
    Physical_Activity_Hours: float
    Health_Issues: str
    Occupation: str
    Smoking: int
    Alcohol_Consumption: int

def predict(input_data: InputData) -> str:
    data_dict = input_data.dict()    
    
    df_input = pd.DataFrame([data_dict])
    
    prediction = champion_model.predict(df_input)
    return prediction[0]

@app.post("/api/v1/predict")
def predict_sleep_quality(input_data: InputData) -> dict:
    try:
        result = predict(input_data)
        return {"prediction": result}
    
    except Exception as e:
        print(f"Error en la predicción: {e}")
        
        raise HTTPException(status_code=500, detail=f"En error occurred while processing the request: {str(e)}")
    


