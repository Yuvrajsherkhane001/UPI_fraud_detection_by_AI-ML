# src/fraud_detector.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class UPIFraudDetector:
    """
    Fraud detection model for UPI transactions
    Trains on Day 1 data, evaluates on Day 2 to simulate concept drift
    """

    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.best_model_name = None
        self.best_model_score = 0

        # Create model directory if it doesn't exist
        import os
        os.makedirs(model_dir, exist_ok=True)

    def load_data(self, day1_path='data/day1_features_engineered.csv',
                  day2_path='data/day2_features_engineered.csv'):
        """Load engineered features for both days"""
        print("Loading engineered datasets...")
        self.day1_df = pd.read_csv(day1_path)
        self.day2_df = pd.read_csv(day2_path)

        print(f"Day 1 shape: {self.day1_df.shape}")
        print(f"Day 2 shape: {self.day2_df.shape}")

        # Define feature columns (exclude non-feature columns)
        exclude_cols = ['transaction_id', 'timestamp', 'user_id', 'merchant_id',
                       'device_id', 'ip_address', 'is_fraud', 'fraud_type',
                       'time_of_day']  # time_of_day is categorical, we have dummies

        self.feature_columns = [col for col in self.day1_df.columns if col not in exclude_cols]
        print(f"Number of features: {len(self.feature_columns)}")

        # Prepare features and labels
        self.X_day1 = self.day1_df[self.feature_columns].fillna(0)
        self.y_day1 = self.day1_df['is_fraud']

        self.X_day2 = self.day2_df[self.feature_columns].fillna(0)
        self.y_day2 = self.day2_df['is_fraud']

        print(f"Day 1 fraud rate: {self.y_day1.mean():.3f}")
        print(f"Day 2 fraud rate: {self.y_day2.mean():.3f}")

        return self.X_day1, self.y_day1, self.X_day2, self.y_day2

    def create_models(self):
        """Initialize the models to train"""
        # Handle class imbalance with scale_pos_weight for XGBoost
        fraud_rate = self.y_day1.mean()
        scale_pos_weight = (1 - fraud_rate) / fraud_rate if fraud_rate > 0 else 1

        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'isolation_forest': IsolationForest(
                contamination=fraud_rate,  # Expected fraud rate
                random_state=42,
                n_jobs=-1
            )
        }

        print(f"Initialized {len(self.models)} models")
        print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")

    def train_models(self):
        """Train all models on Day 1 data"""
        print("\nTraining models on Day 1 data...")
        self.train_results = {}

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            try:
                # For Isolation Forest, we don't use labels during training (unsupervised)
                if name == 'isolation_forest':
                    model.fit(self.X_day1)
                    # For evaluation, we need to predict anomalies (-1 for anomalies, 1 for normal)
                    # Convert to fraud labels: 1 for anomaly (fraud), 0 for normal
                    y_day1_pred = model.predict(self.X_day1)
                    y_day1_pred = np.where(y_day1_pred == -1, 1, 0)
                    y_day1_pred_proba = model.decision_function(self.X_day1)
                    # Convert decision function to probabilities (higher score = more normal)
                    # We want higher probability for fraud, so we invert and scale
                    y_day1_pred_proba = 1 / (1 + np.exp(y_day1_pred_proba))  # Sigmoid
                else:
                    model.fit(self.X_day1, self.y_day1)
                    y_day1_pred = model.predict(self.X_day1)
                    # Get probability estimates for fraud class
                    if hasattr(model, "predict_proba"):
                        y_day1_pred_proba = model.predict_proba(self.X_day1)[:, 1]
                    else:
                        # For models without predict_proba, use decision function
                        y_day1_pred_proba = model.decision_function(self.X_day1)
                        # Normalize to [0,1] range
                        y_day1_pred_proba = (y_day1_pred_proba - y_day1_pred_proba.min()) / \
                                          (y_day1_pred_proba.max() - y_day1_pred_proba.min() + 1e-8)

                # Calculate training metrics
                train_auc = roc_auc_score(self.y_day1, y_day1_pred_proba)
                train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
                    self.y_day1, y_day1_pred, average='binary'
                )

                self.train_results[name] = {
                    'model': model,
                    'train_auc': train_auc,
                    'train_precision': train_precision,
                    'train_recall': train_recall,
                    'train_f1': train_f1,
                    'predictions': y_day1_pred,
                    'probabilities': y_day1_pred_proba
                }

                print(f"  Train AUC: {train_auc:.4f}")
                print(f"  Train Precision: {train_precision:.4f}")
                print(f"  Train Recall: {train_recall:.4f}")
                print(f"  Train F1: {train_f1:.4f}")

            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
                self.train_results[name] = {'error': str(e)}

    def evaluate_on_day2(self):
        """Evaluate all trained models on Day 2 data (concept drift simulation)"""
        print("\nEvaluating models on Day 2 data (concept drift test)...")
        self.eval_results = {}

        for name, result in self.train_results.items():
            if 'error' in result:
                print(f"\n{name}: Skipping evaluation due to training error")
                continue

            print(f"\nEvaluating {name}...")
            model = result['model']

            try:
                # For Isolation Forest, same conversion as in training
                if name == 'isolation_forest':
                    y_day2_pred = model.predict(self.X_day2)
                    y_day2_pred = np.where(y_day2_pred == -1, 1, 0)
                    y_day2_pred_proba = model.decision_function(self.X_day2)
                    y_day2_pred_proba = 1 / (1 + np.exp(y_day2_pred_proba))  # Sigmoid
                else:
                    y_day2_pred = model.predict(self.X_day2)
                    if hasattr(model, "predict_proba"):
                        y_day2_pred_proba = model.predict_proba(self.X_day2)[:, 1]
                    else:
                        y_day2_pred_proba = model.decision_function(self.X_day2)
                        y_day2_pred_proba = (y_day2_pred_proba - y_day2_pred_proba.min()) / \
                                          (y_day2_pred_proba.max() - y_day2_pred_proba.min() + 1e-8)

                # Calculate Day 2 metrics
                day2_auc = roc_auc_score(self.y_day2, y_day2_pred_proba)
                day2_precision, day2_recall, day2_f1, _ = precision_recall_fscore_support(
                    self.y_day2, y_day2_pred, average='binary'
                )

                # Also calculate on Day 1 for comparison (overfitting check)
                day1_auc = roc_auc_score(self.y_day1, result['probabilities'])
                day1_precision, day1_recall, day1_f1, _ = precision_recall_fscore_support(
                    self.y_day1, result['predictions'], average='binary'
                )

                self.eval_results[name] = {
                    'model': model,
                    'day1_auc': day1_auc,
                    'day1_precision': day1_precision,
                    'day1_recall': day1_recall,
                    'day1_f1': day1_f1,
                    'day2_auc': day2_auc,
                    'day2_precision': day2_precision,
                    'day2_recall': day2_recall,
                    'day2_f1': day2_f1,
                    'day2_predictions': y_day2_pred,
                    'day2_probabilities': y_day2_pred_proba,
                    'overfit_auc': day1_auc - day2_auc,  # Positive = overfitting
                    'overfit_f1': day1_f1 - day2_f1
                }

                print(f"  Day 1 AUC: {day1_auc:.4f} | Day 2 AUC: {day2_auc:.4f} | Diff: {day1_auc-day2_auc:.4f}")
                print(f"  Day 1 F1:  {day1_f1:.4f} | Day 2 F1:  {day2_f1:.4f} | Diff: {day1_f1-day2_f1:.4f}")
                print(f"  Day 2 Precision: {day2_precision:.4f}")
                print(f"  Day 2 Recall: {day2_recall:.4f}")

                # Track best model based on Day 2 AUC (most important for concept drift)
                if day2_auc > self.best_model_score:
                    self.best_model_score = day2_auc
                    self.best_model_name = name

            except Exception as e:
                print(f"  Error evaluating {name}: {str(e)}")
                self.eval_results[name] = {'error': str(e)}

    def save_best_model(self):
        """Save the best performing model and its metadata"""
        if self.best_model_name is None:
            print("\nNo valid models to save!")
            return

        print(f"\nSaving best model: {self.best_model_name} (Day 2 AUC: {self.best_model_score:.4f})")

        best_model = self.eval_results[self.best_model_name]['model']
        model_path = f"{self.model_dir}/best_fraud_model.pkl"
        joblib.dump(best_model, model_path)

        # Save feature columns for later use
        feature_path = f"{self.model_dir}/feature_columns.json"
        with open(feature_path, 'w') as f:
            json.dump(self.feature_columns, f)

        # Save model metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'best_model_day2_auc': float(self.best_model_score),
            'training_date': datetime.now().isoformat(),
            'day1_fraud_rate': float(self.y_day1.mean()),
            'day2_fraud_rate': float(self.y_day2.mean()),
            'feature_count': len(self.feature_columns),
            'model_performance': {}
        }

        # Add performance metrics for all models
        for name, result in self.eval_results.items():
            if 'error' not in result:
                metadata['model_performance'][name] = {
                    'day1_auc': float(result['day1_auc']),
                    'day2_auc': float(result['day2_auc']),
                    'day1_f1': float(result['day1_f1']),
                    'day2_f1': float(result['day2_f1']),
                    'day1_precision': float(result['day1_precision']),
                    'day2_precision': float(result['day2_precision']),
                    'day1_recall': float(result['day1_recall']),
                    'day2_recall': float(result['day2_recall']),
                    'overfit_auc': float(result['overfit_auc']),
                    'overfit_f1': float(result['overfit_f1'])
                }

        metadata_path = f"{self.model_dir}/model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to: {model_path}")
        print(f"Feature columns saved to: {feature_path}")
        print(f"Metadata saved to: {metadata_path}")

    def generate_report(self):
        """Generate a comprehensive performance report"""
        print("\n" + "="*60)
        print("UPI FRAUD DETECTION MODEL PERFORMANCE REPORT")
        print("="*60)
        print(f"Training Data: Day 1 transactions ({len(self.X_day1)} samples)")
        print(f"Test Data: Day 2 transactions ({len(self.X_day2)} samples)")
        print(f"Concept Drift Simulation: Train on Day 1, Test on Day 2")
        print("-"*60)

        # Header
        print(f"{'Model':<20} {'Day 1 AUC':<10} {'Day 2 AUC':<10} {'Day 2 F1':<10} {'Day 2 Prec':<10} {'Day 2 Rec':<10}")
        print("-"*60)

        # Sort by Day 2 AUC (descending)
        sorted_models = sorted(
            [(name, result) for name, result in self.eval_results.items() if 'error' not in result],
            key=lambda x: x[1]['day2_auc'],
            reverse=True
        )

        for name, result in sorted_models:
            print(f"{name:<20} {result['day1_auc']:<10.4f} {result['day2_auc']:<10.4f} "
                  f"{result['day2_f1']:<10.4f} {result['day2_precision']:<10.4f} {result['day2_recall']:<10.4f}")

        print("-"*60)
        if self.best_model_name:
            print(f"BEST MODEL (by Day 2 AUC): {self.best_model_name}")
            print(f"Day 2 AUC: {self.best_model_score:.4f}")
        print("="*60)

        # Feature importance for tree-based models
        if self.best_model_name in ['random_forest', 'xgboost']:
            model = self.eval_results[self.best_model_name]['model']
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)

                print("\nTOP 10 MOST IMPORTANT FEATURES:")
                print(feature_importance.head(10))
                print("-"*60)

        return self.eval_results

    def predict_fraud_probability(self, transaction_features):
        """
        Predict fraud probability for a single transaction or batch
        """
        if self.best_model_name is None:
            raise ValueError("No model trained yet. Call train_models() first.")

        model = self.eval_results[self.best_model_name]['model']

        # Ensure we have the right features
        if isinstance(transaction_features, dict):
            # Convert single transaction dict to DataFrame
            df = pd.DataFrame([transaction_features])
        elif isinstance(transaction_features, pd.DataFrame):
            df = transaction_features.copy()
        else:
            # Assume it's already a numpy array or list
            df = pd.DataFrame(transaction_features, columns=self.feature_columns)

        # Select only the features we trained on
        df = df[self.feature_columns].fillna(0)

        # Predict probability
        if hasattr(model, "predict_proba"):
            fraud_prob = model.predict_proba(df)[:, 1]
        else:
            # For models without predict_proba
            decision = model.decision_function(df)
            fraud_prob = 1 / (1 + np.exp(-decision))  # Sigmoid to get probability

        return fraud_prob

def main():
    """Main function to run the fraud detection pipeline"""
    print("UPI FRAUD DETECTION SYSTEM")
    print("="*50)

    # Initialize detector
    detector = UPIFraudDetector(model_dir='models')

    # Load data
    detector.load_data()

    # Create models
    detector.create_models()

    # Train models
    detector.train_models()

    # Evaluate on Day 2 (concept drift test)
    detector.evaluate_on_day2()

    # Generate report
    detector.generate_report()

    # Save best model
    detector.save_best_model()

    print("\nFraud detection pipeline completed!")
    print("Model artifacts saved in 'models/' directory")

    return detector

if __name__ == "__main__":
    main()