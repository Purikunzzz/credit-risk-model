import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['total_late_payment'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] +
        df['NumberOfTime60-89DaysPastDueNotWorse'] +
        df['NumberOfTimes90DaysLate']
    )
    
    df['monthly_debt'] = df['DebtRatio'] * df['MonthlyIncome']
    
    df['income_to_debt'] = df['MonthlyIncome'] / (df['monthly_debt'] + 1)
    
    return df