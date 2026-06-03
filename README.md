---
title: Employee Attrition Predictor
emoji: 👔
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Employee Attrition Predictor API

ML-powered REST API that predicts whether an employee is likely to leave a company using a Random Forest classifier trained on the IBM HR Analytics dataset (1,470 records).

## Tech Stack
- **Model:** Random Forest Classifier (86% accuracy)
- **API:** FastAPI
- **Deployment:** Docker + GitHub Actions CI/CD + HuggingFace Spaces

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API status |
| `/predict` | POST | Predict attrition |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |

## Input Features
| Feature | Type | Description |
|---|---|---|
| Age | int | Employee age |
| MonthlyIncome | float | Monthly salary |
| YearsAtCompany | int | Years at company |
| JobSatisfaction | int | 1-4 (1=Low, 4=High) |
| WorkLifeBalance | int | 1-4 (1=Bad, 4=Best) |
| OverTime | int | 0=No, 1=Yes |
| EnvironmentSatisfaction | int | 1-4 |
| YearsSinceLastPromotion | int | Years since promotion |
| NumCompaniesWorked | int | Previous companies |
| DistanceFromHome | int | Distance in km |

## Example Request
```json
{
  "Age": 35,
  "MonthlyIncome": 45000,
  "YearsAtCompany": 2,
  "JobSatisfaction": 1,
  "WorkLifeBalance": 1,
  "OverTime": 1,
  "EnvironmentSatisfaction": 1,
  "YearsSinceLastPromotion": 0,
  "NumCompaniesWorked": 5,
  "DistanceFromHome": 25
}
```

## Example Response
```json
{
  "attrition_prediction": "Will Leave",
  "attrition_probability": 0.79,
  "risk_level": "High Risk",
  "recommendation": "Immediate HR intervention suggested"
}
```

## Author
**Muhammad Zeeshan** — [GitHub](https://github.com/zeeshanmemon923-ux)