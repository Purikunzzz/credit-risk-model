## Credit Risk Prediction API

A machine learning API that predicts the probability of a customer 
defaulting on a loan within 2 years. 

## Insight
During EDA, I found that `RevolvingUtilizationOfUnsecuredLines` 
and `DebtRatio` had extreme outliers (max values of 50,000+ and 300,000+) 
despite being ratio features. I capped them using domain knowledge and 
95th percentile instead of dropping rows, to preserve data integrity.

Feature importance revealed that `total_late_payment` (a feature I 
engineered by combining 3 late payment columns) became the single most 
important feature, outperforming all original features.

# What I learned
Every step in the Machine Learning pipeline matters. Good data cleaning is like 
selecting quality parts for a car — the foundation everything else 
depends on. Choosing the right algorithm is like assembling those parts 
into something that runs. But feature engineering is the real tuning — 
it's what makes the difference between a car that runs and one that wins. (It's my first project btw ;))

## Dataset
[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — 150,000 loan records from Kaggle.

## Results
| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.8117 |
| Random Forest | 0.8407 |
| XGBoost (tuned) | 0.8693 |

## Project Structure
credit-risk-model/
├── notebooks/ # EDA, baseline models, tuning, feature engineering
├── src/ # preprocessing pipeline
├── api/ # FastAPI application
├── models/ # saved model (.joblib)
└── data/ # raw and processed data (not tracked in git)

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle and place in data/raw/

# Run API
uvicorn api.main:app --reload
```

## API Usage
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.75,
    "age": 35,
    "NumberOfTime30_59DaysPastDueNotWorse": 2,
    "DebtRatio": 0.5,
    "MonthlyIncome": 5000.0,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 1,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60_89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2,
    "MonthlyIncome_missing": 0
  }'
```

Response:
```json
{
  "default_probability": 0.8835,
  "risk_level": "High"
}
```

## Tech Stack
- Python, XGBoost, scikit-learn, pandas
- FastAPI, uvicorn, joblib

