import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI()

class PredictionInput(BaseModel):
    age: int
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    gender: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

# Model file check karne ke liye safe code
MODEL_PATH = "svc_model_3.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    raise FileNotFoundError(f"Missing model file! Please ensure {MODEL_PATH} is uploaded.")

@app.get("/")
def home():
    return {"status": "Backend is running live!"}

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        data_dict = {
            "age": [data.age],
            "workclass": [data.workclass.strip()],
            "education": [data.education.strip()],
            "marital-status": [data.marital_status.strip()],
            "occupation": [data.occupation.strip()],
            "relationship": [data.relationship.strip()],
            "gender": [data.gender.strip()],
            "capital-gain": [data.capital_gain],
            "capital-loss": [data.capital_loss],
            "hours-per-week": [data.hours_per_week],
            "native-country": [data.native_country.strip()]
        }
        df = pd.DataFrame(data_dict)
        
        # Prediction nikaal kar pehla element extracted kiya taake array array error na aaye
        raw_prediction = model.predict(df)
        prediction_val = raw_prediction[0]
        
        return {"prediction": str(prediction_val)}
        
    except Exception as e:
        return {"error": str(e), "message": "Pipeline processing failed inside container"}

if __name__ == "__main__":
    # Hugging face port default specification block
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
