from pydantic import BaseModel, Field, ConfigDict

class PredictRequest(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ..., 
        ge=0.0, 
        description="Total balance on revolving credit cards and personal lines of credit divided by the sum of credit limits."
    )
    age: float = Field(
        ..., 
        ge=0.0, 
        description="Age of borrower in years."
    )
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ..., 
        alias="NumberOfTime30-59DaysPastDueNotWorse", 
        ge=0, 
        description="Number of times borrower has been 30-59 days past due but no worse in the last 2 years."
    )
    DebtRatio: float = Field(
        ..., 
        ge=0.0, 
        description="Monthly debt payments, alimony, living costs divided by monthly gross income."
    )
    MonthlyIncome: float = Field(
        ..., 
        ge=0.0, 
        description="Monthly gross income."
    )
    NumberOfOpenCreditLinesAndLoans: int = Field(
        ..., 
        ge=0, 
        description="Number of open credit loans (like auto loan or mortgage) and lines of credit."
    )
    NumberOfTimes90DaysLate: int = Field(
        ..., 
        ge=0, 
        description="Number of times borrower has been 90 days or more past due."
    )
    NumberRealEstateLoansOrLines: int = Field(
        ..., 
        ge=0, 
        description="Number of mortgage and real estate loans including home equity lines of credit."
    )
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ..., 
        alias="NumberOfTime60-89DaysPastDueNotWorse", 
        ge=0, 
        description="Number of times borrower has been 60-89 days past due but no worse in the last 2 years."
    )
    NumberOfDependents: int = Field(
        ..., 
        ge=0, 
        description="Number of dependents in family excluding themselves."
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "RevolvingUtilizationOfUnsecuredLines": 0.0152,
                "age": 45.0,
                "NumberOfTime30-59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.12,
                "MonthlyIncome": 6000.0,
                "NumberOfOpenCreditLinesAndLoans": 8,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 1,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2
            }
        }
    )

class PredictResponse(BaseModel):
    default_probability: float = Field(..., description="Calibrated probability of default (isotonic calibration).")
    raw_probability: float = Field(..., description="Raw uncalibrated probability of default (from XGBoost pipeline).")
    threshold: float = Field(..., description="Decision classification threshold.")
    prediction: int = Field(..., description="Binary prediction (0 for LOW RISK, 1 for HIGH RISK).")
    risk_level: str = Field(..., description="Qualitative risk classification level assessment.")
