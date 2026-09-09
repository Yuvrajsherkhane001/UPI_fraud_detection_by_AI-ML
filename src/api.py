# src/api.py
"""
FastAPI service for UPI fraud detection
Loads trained model and provides real-time fraud scoring
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
import os
import subprocess
import sys
from typing import Dict, Any
import uvicorn
from feature_engineer import UPIFeatureEngineer

# Initialize FastAPI app
app = FastAPI(
    title="UPI Fraud Detection API",
    description="Real-time fraud scoring for UPI transactions",
    version="1.0.0"
)

# Global variables for model and feature engineer
model = None
feature_engineer = None
feature_columns = None
model_metadata = None

class TransactionRequest(BaseModel):
    """Request model for transaction fraud scoring"""
    transaction_id: str
    timestamp: str  # ISO format: "2026-09-08T14:30:00"
    user_id: str
    merchant_id: str
    amount: float
    device_id: str
    ip_address: str

class FraudScoreResponse(BaseModel):
    """Response model for fraud scoring"""
    transaction_id: str
    fraud_probability: float
    fraud_prediction: bool  # True if fraud_probability > 0.5
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    model_version: str
    timestamp: str

@app.on_event("startup")
async def load_model():
    """Load the trained model and feature engineering components on startup"""
    global model, feature_engineer, feature_columns, model_metadata

    # Step 1: Check if model exists, if not generate and train
    model_path = "models/best_fraud_model.pkl"
    if not os.path.exists(model_path):
        print("[INFO] Model file not found. Generating data and training model...")
        try:
            # Generate data
            print("[INFO] Running data generation...")
            result = subprocess.run([sys.executable, "src/data_generator.py"],
                                  capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"[ERROR] Data generation failed: {result.stderr}")
                raise RuntimeError("Data generation failed")
            print("[INFO] Data generation completed.")
        except subprocess.TimeoutExpired:
            print("[ERROR] Data generation timed out")
            raise RuntimeError("Data generation timed out")
        except Exception as e:
            print(f"[ERROR] Data generation failed: {e}")
            raise RuntimeError(f"Data generation failed: {e}")

        try:
            # Train model
            print("[INFO] Running model training...")
            result = subprocess.run([sys.executable, "src/fraud_detector.py"],
                                  capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"[ERROR] Model training failed: {result.stderr}")
                raise RuntimeError("Model training failed")
            print("[INFO] Model training completed.")
        except subprocess.TimeoutExpired:
            print("[ERROR] Model training timed out")
            raise RuntimeError("Model training timed out")
        except Exception as e:
            print(f"[ERROR] Model training failed: {e}")
            raise RuntimeError(f"Model training failed: {e}")

    # Step 2: Load the model and related files (should exist now)
    try:
        # Load the best model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found after training: {model_path}")

        model = joblib.load(model_path)
        print(f"[OK] Model loaded from {model_path}")

        # Load feature columns
        feature_path = "models/feature_columns.json"
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature columns file not found: {feature_path}")

        with open(feature_path, 'r') as f:
            feature_columns = json.load(f)
        print(f"[OK] Feature columns loaded ({len(feature_columns)} features)")

        # Load model metadata
        metadata_path = "models/model_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            print(f"[OK] Model metadata loaded: {model_metadata.get('best_model_name', 'Unknown')}")
        else:
            model_metadata = {"best_model_name": "Unknown", "training_date": "Unknown"}
            print("[WARN] Model metadata not found")

        # Initialize feature engineer
        feature_engineer = UPIFeatureEngineer()
        print("[OK] Feature engineer initialized")

    except Exception as e:
        print(f"[ERROR] Error loading model: {str(e)}")
        # Don't fail startup - allow health checks to work
        # Model will remain None, health check will fail
        pass

def preprocess_transaction(transaction: TransactionRequest) -> pd.DataFrame:
    """
    Convert a transaction request to the format expected by our feature engineering pipeline
    """
    # Convert to DataFrame (single row)
    df = pd.DataFrame([transaction.dict()])

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Add placeholder for is_fraud (required by feature engineering but not known for prediction)
    df['is_fraud'] = 0  # Placeholder
    df['fraud_type'] = 'none'  # Placeholder

    return df

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "UPI Fraud Detection API",
        "version": "1.0.0",
        "status": "operational" if model is not None else "model_not_loaded",
        "model_info": model_metadata
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_engineer_ready": feature_engineer is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.post("/score_transaction", response_model=FraudScoreResponse)
async def score_transaction(transaction: TransactionRequest):
    """
    Score a single UPI transaction for fraud probability
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if feature_engineer is None:
        raise HTTPException(status_code=503, detail="Feature engineer not initialized")

    try:
        # Step 1: Preprocess the transaction
        df_raw = preprocess_transaction(transaction)

        # Step 2: Apply feature engineering
        # Note: We set fit=False since we're using the same feature engineering logic
        # In a production system, you'd save/load a pre-fitted feature engineer
        df_features, _ = feature_engineer.create_features(df_raw.copy(), fit=False)

        # Step 3: Select only the features used during training
        if feature_columns is None:
            raise HTTPException(status_code=500, detail="Feature columns not loaded")

        # Ensure we have all required features
        missing_features = set(feature_columns) - set(df_features.columns)
        if missing_features:
            # Fill missing features with 0 (or appropriate defaults)
            for feat in missing_features:
                df_features[feat] = 0

        # Select features in the correct order
        X = df_features[feature_columns].fillna(0)

        # Step 4: Get fraud probability from model
        fraud_probability = model.predict_proba(X)[:, 1][0]

        # Step 5: Determine prediction and risk level
        fraud_prediction = fraud_probability > 0.5

        if fraud_probability < 0.3:
            risk_level = "LOW"
        elif fraud_probability < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        # Step 6: Prepare response
        response = FraudScoreResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability=float(fraud_probability),
            fraud_prediction=bool(fraud_prediction),
            risk_level=risk_level,
            model_version=model_metadata.get('best_model_name', 'Unknown'),
            timestamp=pd.Timestamp.now().isoformat()
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring transaction: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not available")

    return model_metadata
