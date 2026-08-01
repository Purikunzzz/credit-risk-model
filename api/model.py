import joblib
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import engineer_features

model = joblib.load('models/xgboost.joblib')

def predict_default(data: dict) -> dict:
    df = pd.DataFrame([data])

    # change column name from - to _
    df = df.rename(columns={
        'NumberOfTime30_59DaysPastDueNotWorse': 'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTime60_89DaysPastDueNotWorse': 'NumberOfTime60-89DaysPastDueNotWorse'
    })

    # feature engineering
    df = engineer_features(df)

    # predict
    prob = model.predict_proba(df)[0][1]

    if prob < 0.1:
        risk_level = 'Low'
    elif prob < 0.3:
        risk_level = 'Medium'
    else:
        risk_level = 'High'

    return {
        'default_probability': round(float(prob), 4),
        'risk_level': risk_level
    }