from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load model
with open('app/model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI(
    title="Employee Attrition Predictor",
    description="Predicts whether an employee is likely to leave the company using ML",
    version="1.0.0"
)

# Input schema
class EmployeeData(BaseModel):
    Age: int
    MonthlyIncome: float
    YearsAtCompany: int
    JobSatisfaction: int        # 1-4 (1=Low, 4=High)
    WorkLifeBalance: int        # 1-4 (1=Bad, 4=Best)
    OverTime: int               # 0=No, 1=Yes
    EnvironmentSatisfaction: int # 1-4
    YearsSinceLastPromotion: int
    NumCompaniesWorked: int
    DistanceFromHome: int

@app.get("/")
def home():
    return {
        "message": "Employee Attrition Predictor API",
        "status": "running",
        "author": "Muhammad Zeeshan",
        "github": "https://github.com/zeeshanmemon923-ux"
    }

@app.post("/predict")
def predict(data: EmployeeData):
    features = np.array([[
        data.Age,
        data.MonthlyIncome,
        data.YearsAtCompany,
        data.JobSatisfaction,
        data.WorkLifeBalance,
        data.OverTime,
        data.EnvironmentSatisfaction,
        data.YearsSinceLastPromotion,
        data.NumCompaniesWorked,
        data.DistanceFromHome
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    # Risk level
    if probability >= 0.7:
        risk = "High Risk"
        recommendation = "Immediate HR intervention suggested"
    elif probability >= 0.4:
        risk = "Medium Risk"
        recommendation = "Schedule check-in with manager"
    else:
        risk = "Low Risk"
        recommendation = "Employee likely to stay"

    return {
        "attrition_prediction": "Will Leave" if prediction == 1 else "Will Stay",
        "attrition_probability": round(float(probability), 2),
        "risk_level": risk,
        "recommendation": recommendation
    }

@app.get("/health")
def health():
    return {"status": "healthy"}