import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies exact feature-engineering formulas from the ML project pipeline:
    1. TotalDelinquency = sum of three past-due columns
    2. HasDelinquency = 1 if TotalDelinquency > 0, else 0
    3. HasSevereDelinquency = 1 if NumberOfTimes90DaysLate > 0, else 0
    4. LogCreditUtilization = log1p(RevolvingUtilizationOfUnsecuredLines)
    5. HighCreditUtilization = 1 if RevolvingUtilizationOfUnsecuredLines > 1, else 0
    6. LogMonthlyIncome = log1p(MonthlyIncome)
    7. IncomeCategory = cut(MonthlyIncome) into Low, Lower-Middle, Upper-Middle, High
    8. LogDebtRatio = log1p(DebtRatio)
    9. HighDebtRatio = 1 if DebtRatio > 1, else 0
    10. TotalCreditAccounts = Open lines + Real estate loans
    """
    fe_df = df.copy()
    
    # 1. TotalDelinquency
    fe_df["TotalDelinquency"] = (
        fe_df["NumberOfTime30-59DaysPastDueNotWorse"]
        + fe_df["NumberOfTime60-89DaysPastDueNotWorse"]
        + fe_df["NumberOfTimes90DaysLate"]
    )
    
    # 2. HasDelinquency
    fe_df["HasDelinquency"] = (fe_df["TotalDelinquency"] > 0).astype(int)
    
    # 3. HasSevereDelinquency
    fe_df["HasSevereDelinquency"] = (fe_df["NumberOfTimes90DaysLate"] > 0).astype(int)
    
    # 4. LogCreditUtilization
    fe_df["LogCreditUtilization"] = np.log1p(fe_df["RevolvingUtilizationOfUnsecuredLines"])
    
    # 5. HighCreditUtilization
    fe_df["HighCreditUtilization"] = (fe_df["RevolvingUtilizationOfUnsecuredLines"] > 1).astype(int)
    
    # 6. LogMonthlyIncome
    fe_df["LogMonthlyIncome"] = np.log1p(fe_df["MonthlyIncome"])
    
    # 7. IncomeCategory
    fe_df["IncomeCategory"] = pd.cut(
        fe_df["MonthlyIncome"],
        bins=[-1, 3400, 5400, 8250, float("inf")],
        labels=["Low", "Lower-Middle", "Upper-Middle", "High"]
    )
    
    # 8. LogDebtRatio
    fe_df["LogDebtRatio"] = np.log1p(fe_df["DebtRatio"])
    
    # 9. HighDebtRatio
    fe_df["HighDebtRatio"] = (fe_df["DebtRatio"] > 1).astype(int)
    
    # 10. TotalCreditAccounts
    fe_df["TotalCreditAccounts"] = (
        fe_df["NumberOfOpenCreditLinesAndLoans"]
        + fe_df["NumberRealEstateLoansOrLines"]
    )
    
    return fe_df
