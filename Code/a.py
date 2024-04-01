from fastapi import FastAPI, HTTPException
import uvicorn
import pickle
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel

# Assuming the model and scaler are trained and saved separately
file_model = open('predictive_maintenance_model.pkl', 'rb')
model = pickle.load(file_model)
file_model.close()

file_scaler = open('scaler.pkl', 'rb')
scaler = pickle.load(file_scaler)
file_scaler.close()

app = FastAPI()

class EquipmentData(BaseModel):
    temperature: float
    vibration: float
    noise_level: float
    hours_operated: float

@app.post("/predict_failure")
async def predict_failure(data: EquipmentData):
    numerical_data = [data.temperature, data.vibration, data.noise_level, data.hours_operated]
    
    # Scaling the input data
    scaled_data = scaler.transform([numerical_data])
    
    # Predicting equipment failure (1 for failure, 0 for no failure)
    prediction = model.predict(scaled_data)[0]
    
    # Assuming the model outputs a binary classification: 1 for likely failure, 0 for unlikely
    if prediction == 1:
        maintenance_recommendation = "Maintenance recommended"
    else:
        maintenance_recommendation = "No maintenance needed"

    return {"maintenance_recommendation": maintenance_recommendation}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")
