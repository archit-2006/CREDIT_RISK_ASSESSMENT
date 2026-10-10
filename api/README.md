# Credit Risk Assessment API

A production-ready FastAPI backend serving the calibrated **Frozen XGBoost Model 1** pipeline to predict borrower default risk. The API accepts only raw borrower features, automatically applies the complete 20-feature engineering pipeline used during training, scales/encodes inputs, and returns the classification decision based on the calibrated probability and optimal threshold of **0.21**.

## Features
- **Automatic Feature Engineering**: Reproduces the full notebook feature engineering logic (delinquency aggregations, log transforms, ratio indicators, and income categorization).
- **Startup Loading**: Loads trained artifacts once on startup directly from `final_artifacts/` (`model_1_preprocessor.joblib`, `model_1_xgboost.joblib`, `model_1_isotonic_calibrator.joblib`, `model_1_artifact_info.joblib`).
- **Isotonic Probability Calibration**: Calibrates raw XGBoost probabilities into empirically accurate default risk probabilities.
- **Robust Schema Validation**: Pydantic v2 schemas reject invalid inputs (e.g., negative age, negative income, negative credit lines).
- **CORS Middleware**: Preconfigured for communication with frontend dashboards.
- **Interactive Documentation**: Swagger UI documentation out-of-the-box.

---

## Getting Started

### 1. Install Dependencies
Run the following command inside your virtual environment to install requirements:
```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI Server
Run `uvicorn` from the `api` folder:
```bash
uvicorn main:app --reload
```
By default, the server runs on `http://127.0.0.1:8000`.

### 3. Interactive Swagger Documentation
Access the interactive API documentation and test endpoints at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## API Endpoints

### 1. Root Check
- **Endpoint**: `GET /`
- **Description**: Verifies that the API service is running.
- **Response**:
  ```json
  {
      "message": "Credit Risk Assessment API is running"
  }
  ```

### 2. Health & Model Details
- **Endpoint**: `GET /health`
- **Description**: Returns model configuration, threshold status, and calibration method.
- **Response**:
  ```json
  {
      "status": "healthy",
      "model": "Frozen XGBoost Model 1",
      "threshold": 0.21,
      "calibration": "isotonic"
  }
  ```

### 3. Predict Credit Risk
- **Endpoint**: `POST /predict`
- **Description**: Evaluates credit default probability and classifies the risk profile.
- **Request Body (JSON)**:
  ```json
  {
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
  ```
- **Response Body (JSON)**:
  ```json
  {
      "default_probability": 0.0067,
      "raw_probability": 0.0077,
      "threshold": 0.21,
      "prediction": 0,
      "risk_level": "LOW RISK"
  }
  ```

---

## Risk Classifications and Thresholds

- **Model Decision Threshold**: `0.21` (optimized on out-of-fold calibrated probabilities to maximize F1 score).
- **Low Risk (`prediction = 0`)**: The model calculates a calibrated default probability strictly less than `0.21` (`< 21%`). The borrower qualifies as a low default hazard.
- **High Risk (`prediction = 1`)**: The model calculates a calibrated default probability equal to or greater than `0.21` (`≥ 21%`). The borrower represents an elevated default risk according to lending risk limits.

---

## Running Automated Tests
Run the standalone test script to verify endpoint functionality:
```bash
python test_api.py
```
This verifies endpoint status codes, schema consistency, probability calculations, and input validation rejection.
