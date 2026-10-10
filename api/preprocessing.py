import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies exact feature-engineering formulas from the ML project pipeline:
    Generates all 10 domain-engineered features (TotalDelinquency, HasDelinquency,
    HasSevereDelinquency, LogCreditUtilization, HighCreditUtilization, LogMonthlyIncome,
    IncomeCategory, LogDebtRatio, HighDebtRatio, TotalCreditAccounts).
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
