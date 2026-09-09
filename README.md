# UPI Fraud Detection System

A machine learning-based system for detecting fraudulent UPI (Unified Payments Interface) transactions, designed for hackathon submissions.

## Project Overview

This system detects fraudulent UPI transactions by:
1. Generating synthetic UPI transaction datasets with realistic fraud patterns
2. Engineering comprehensive features from transaction data
3. Training multiple machine learning models to detect fraud
4. Evaluating models for concept drift (training on Day 1, testing on Day 2)
5. Providing a real-time fraud scoring API

## Features

- **Synthetic Data Generation**: Creates realistic Day 1 (clear fraud patterns) and Day 2 (evolved fraud patterns) datasets
- **Feature Engineering**: 51 features including temporal, amount-based, user/merchant/device behavior, velocity metrics, and risk indicators
- **Multiple Models**: Logistic Regression, Random Forest, XGBoost, and Isolation Forest
- **Concept Drift Handling**: Explicitly trained on Day 1 and tested on Day 2 to simulate real-world fraud evolution
- **Model Explainability**: Feature importance analysis for interpretable results
- **Real-time API**: FastAPI service for real-time fraud scoring of transactions
- **Docker-ready**: Easy to containerize and deploy

## Project Structure

```
upi-fraud-detection-hackathon/
├── data/                          # Generated datasets (gitignored)
│   ├── day1_transactions.csv      # Day 1 UPI transactions
│   ├── day2_transactions.csv      # Day 2 UPI transactions
│   ├── day1_features_engineered.csv  # Engineered features for Day 1
│   └── day2_features_engineered.csv  # Engineered features for Day 2
├── models/                        # Trained models and metadata (gitignored)
│   ├── best_fraud_model.pkl       # Best performing model (Random Forest)
│   ├── feature_columns.json       # Feature column names
│   └── model_metadata.json        # Model performance metadata
├── notebooks/                     # Jupyter notebooks for exploration
├── src/                           # Source code
│   ├── data_generator.py          # Synthetic data generation
│   ├── feature_engineer.py        # Feature engineering pipeline
│   ├── fraud_detector.py          # Model training and evaluation
│   ├── api.py                     # FastAPI fraud scoring service
│   ├── test_fe.py                 # Feature engineering tests
│   ├── test_api.py                # API testing script
│   └── ...                        # Other utility files
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Installation

1. **Clone the repository** (if applicable)
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Generate Data (if needed)
The data has already been generated and committed to the repository. To regenerate:
```bash
python src/data_generator.py
```

### 2. Train Models
```bash
python src/fraud_detector.py
```
This will:
- Load the engineered features
- Train multiple models on Day 1 data
- Evaluate them on Day 2 data (concept drift test)
- Save the best model and metadata to the `models/` directory

### 3. Start the API Service
```bash
python src/api.py
```
<br>markdown\nWhen running locally, the API will be available at `http://localhost:8000`.\nWhen deployed to Render.com, the API will be available at your Render service URL (e.g., `https://your-service.onrender.com`).\n

### 4. Test the API
```bash
python src/test_api.py
```
This will test the API endpoints including:
- Health check
- Model information
- Transaction scoring (normal and high-risk examples)

## API Endpoints

- **GET /** - API information and status
- **GET /health** - Health check endpoint
- **GET /model/info** - Information about the loaded model
- **POST /score_transaction** - Score a single UPI transaction for fraud probability

### Transaction Scoring Example

Request:
```json
{
  "transaction_id": "TEST_TXN_001",
  "timestamp": "2026-09-08T14:30:00",
  "user_id": "USER_9999",
  "merchant_id": "MERCH_999",
  "amount": 5000.0,
  "device_id": "DEV_999",
  "ip_address": "192.168.1.100"
}
```

Response:
```json
{
  "transaction_id": "TEST_TXN_001",
  "fraud_probability": 0.1234,
  "fraud_prediction": false,
  "risk_level": "LOW",
  "model_version": "random_forest",
  "timestamp": "2026-09-09T10:30:00.123456"
}
```

## Model Performance

Based on our evaluation (training on Day 1, testing on Day 2 to simulate concept drift):

| Model | Day 1 AUC | Day 2 AUC | Day 2 F1 | Day 2 Precision | Day 2 Recall |
|-------|-----------|-----------|----------|-----------------|--------------|
| Random Forest | 1.0000 | 0.9372 | 0.7477 | 1.0000 | 0.5971 |
| XGBoost | 1.0000 | 0.9362 | 0.7479 | 0.9980 | 0.5980 |
| Logistic Regression | 0.9990 | 0.8308 | 0.5916 | 0.4509 | 0.8599 |
| Isolation Forest | 0.7567 | 0.6269 | 0.5133 | 0.3792 | 0.7944 |

**Best Model**: Random Forest with 93.72% AUC on Day 2 data

## Key Features for Fraud Detection

The top 10 most important features identified by the Random Forest model:
1. `amount_log` - Log-transformed transaction amount
2. `amount` - Raw transaction amount
3. `amount_vs_user_median` - Transaction amount vs user's median amount
4. `amount_vs_user_avg` - Transaction amount vs user's average amount
5. `user_recent_fraud_rate_24h` - User's fraud rate in last 24 hours
6. `user_fraud_rate` - User's overall fraud rate
7. `user_max_amount` - Maximum transaction amount for user
8. `user_recent_avg_amount_24h` - User's average transaction amount in last 24 hours
9. `user_avg_amount` - User's average transaction amount
10. `user_fraud_count` - Total number of fraudulent transactions for user

## Deployment Options

For hackathon demonstration:
1. **Local Demo**: Run the API locally and use curl or Postman to test
2. **Streamlit Sharing**: Create a simple Streamlit interface that calls the API
3. **Render.com**: Deploy the FastAPI service for a public URL
4. **Hugging Face Spaces**: Deploy with Gradio or Streamlit interface

## Next Steps for Enhancement

1. **Add SHAP values** for individual transaction explanations
2. **Implement streaming** for real-time transaction processing
3. **Add dashboard** for visualizing fraud trends and model performance
4. **Implement model retraining** schedule for concept drift adaptation
5. **Add A/B testing** framework for model comparison

## Hackathon Presentation Tips

1. **Problem Statement**: Clearly explain the UPI fraud detection challenge and concept drift
2. **Data Approach**: Show synthetic data generation process and fraud patterns
3. **Feature Engineering**: Highlight the 51 engineered features and their rationale
4. **Model Selection**: Explain why Random Forest performed best for concept drift
5. **Live Demo**: Show the API scoring transactions in real-time
6. **Impact**: Discuss how this system could reduce fraud losses in real UPI systems

## Notes

- The `.gitignore` file excludes large data and model files to keep the repository lightweight
- All models are saved in the `models/` directory and can be easily loaded for inference
- The system is designed to be extensible for additional features and models
- For production use, consider adding:
  - Input validation and sanitization
  - API authentication and rate limiting
  - Model monitoring and drift detection
  - Batch processing capabilities
