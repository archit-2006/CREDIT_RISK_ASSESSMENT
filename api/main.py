import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from schemas import PredictRequest, PredictResponse
from preprocessing import engineer_features

# Calculate absolute paths relative to main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR_CANDIDATES = [
    os.path.abspath(os.path.join(BASE_DIR, "..", "final_artifacts")),
    os.path.abspath(os.path.join(BASE_DIR, "..", "Notebooks", "Model", "final_artifacts")),
    os.path.abspath(os.path.join(BASE_DIR, "final_artifacts")),
]

ARTIFACTS_DIR = None
for candidate in ARTIFACTS_DIR_CANDIDATES:
    if os.path.exists(candidate):
        ARTIFACTS_DIR = candidate
        break

# Load complete final Model 1 artifacts on startup
try:
    if ARTIFACTS_DIR is None:
        raise FileNotFoundError(f"final_artifacts directory not found in candidates: {ARTIFACTS_DIR_CANDIDATES}")

    preprocessor_path = os.path.join(ARTIFACTS_DIR, "model_1_preprocessor.joblib")
    model_path = os.path.join(ARTIFACTS_DIR, "model_1_xgboost.joblib")
    calibrator_path = os.path.join(ARTIFACTS_DIR, "model_1_isotonic_calibrator.joblib")
    info_path = os.path.join(ARTIFACTS_DIR, "model_1_artifact_info.joblib")

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    calibrator = joblib.load(calibrator_path)
    artifact_info = joblib.load(info_path) if os.path.exists(info_path) else {}

    threshold = float(artifact_info.get("threshold", 0.21))

    print(f"SUCCESS: Loaded Model 1 artifacts from {ARTIFACTS_DIR}")
    print(f"  Model name: {artifact_info.get('model_name', 'Frozen XGBoost Model 1')}")
    print(f"  Threshold: {threshold}")
    print(f"  Calibrator: {type(calibrator).__name__}")
    print(f"  Preprocessor: {type(preprocessor).__name__}")
except Exception as e:
    print(f"CRITICAL: Failed to load Model 1 artifacts: {e}")
    preprocessor = None
    model = None
    calibrator = None
    artifact_info = {}
    threshold = 0.21

app = FastAPI(
    title="Credit Risk Assessment API",
    description="Production-ready FastAPI backend serving the calibrated Frozen XGBoost Model 1.",
    version="3.0"
)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Credit Risk Assessment API is running"}

@app.get("/health")
def health_check():
    if model is None or calibrator is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model package components failed to load on startup"
        )
    return {
        "status": "healthy",
        "model": artifact_info.get("model_name", "Frozen XGBoost Model 1"),
        "threshold": threshold,
        "calibration": artifact_info.get("calibration", "isotonic")
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None or calibrator is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is currently unavailable (model package components not loaded)"
        )
    
    try:
        # Convert incoming data to dictionary, matching aliases (keys with dashes)
        input_dict = request.model_dump(by_alias=True)
        
        # Create single-row DataFrame
        df_raw = pd.DataFrame([input_dict])
        
        # Apply exact feature-engineering formulas (all 20 features)
        df_engineered = engineer_features(df_raw)
        
        # Preprocess features using the trained ColumnTransformer
        X_transformed = preprocessor.transform(df_engineered)
        
        # Predict raw default probability from the XGBoost model
        raw_prob = float(model.predict_proba(X_transformed)[:, 1][0])
        
        # Apply Isotonic Calibration
        calibrated_prob = float(calibrator.predict([raw_prob])[0])
        
        # Determine classification decision using calibrated probability and threshold
        prediction = 1 if calibrated_prob >= threshold else 0
        risk_level = "HIGH RISK" if prediction == 1 else "LOW RISK"
        
        return PredictResponse(
            default_probability=round(calibrated_prob, 4),
            raw_probability=round(raw_prob, 4),
            threshold=round(threshold, 4),
            prediction=prediction,
            risk_level=risk_level
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during feature engineering or model prediction: {str(e)}"
        )