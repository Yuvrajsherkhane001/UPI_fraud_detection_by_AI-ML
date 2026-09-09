# src/feature_engineer.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class UPIFeatureEngineer:
    """
    Feature engineering pipeline for UPI transaction fraud detection
    Creates user, merchant, device, IP, and temporal features
    """

    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.merchant_encoder = LabelEncoder()
        self.device_encoder = LabelEncoder()
        self.amount_scaler = StandardScaler()
        self.fitted = False

    def extract_temporal_features(self, df):
        """Extract time-based features from timestamp"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Basic time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Time of day categories
        df['time_of_day'] = pd.cut(df['hour'],
                                  bins=[0, 6, 12, 18, 24],
                                  labels=['night', 'morning', 'afternoon', 'evening'],
                                  include_lowest=True)

        return df

    def aggregate_user_features(self, df, time_window_hours=24):
        """Create user-level aggregated features"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by user and timestamp for rolling windows
        df = df.sort_values(['user_id', 'timestamp'])

        # User-level aggregations
        user_features = df.groupby('user_id').agg({
            'amount': ['count', 'sum', 'mean', 'std', 'min', 'max'],
            'is_fraud': ['mean', 'sum'],  # historical fraud rate
        }).reset_index()

        # Flatten column names
        user_features.columns = ['user_id'] + [
            f'user_txn_count', f'user_total_amount', f'user_avg_amount',
            f'user_std_amount', f'user_min_amount', f'user_max_amount',
            'user_fraud_rate', 'user_fraud_count'
        ]

        # Handle NaN in std (when only 1 transaction)
        user_features['user_std_amount'] = user_features['user_std_amount'].fillna(0)

        # Recent behavior (last 24 hours)
        cutoff_time = df['timestamp'].max() - timedelta(hours=time_window_hours)
        recent_df = df[df['timestamp'] >= cutoff_time]

        recent_user_features = recent_df.groupby('user_id').agg({
            'amount': ['count', 'sum', 'mean'],
            'is_fraud': ['mean']
        }).reset_index()

        recent_user_features.columns = ['user_id'] + [
            f'user_recent_txn_count_{time_window_hours}h',
            f'user_recent_total_amount_{time_window_hours}h',
            f'user_recent_avg_amount_{time_window_hours}h',
            f'user_recent_fraud_rate_{time_window_hours}h'
        ]

        # Merge features
        user_features = user_features.merge(recent_user_features, on='user_id', how='left')
        user_features = user_features.fillna(0)  # Fill NaN for users with no recent activity

        return user_features

    def aggregate_merchant_features(self, df):
        """Create merchant-level aggregated features"""
        df = df.copy()

        # Merchant-level aggregations
        merchant_features = df.groupby('merchant_id').agg({
            'amount': ['count', 'sum', 'mean', 'std'],
            'is_fraud': ['mean', 'sum'],
            'user_id': 'nunique'  # number of unique users
        }).reset_index()

        # Flatten column names
        merchant_features.columns = ['merchant_id'] + [
            f'merchant_txn_count', f'merchant_total_amount', f'merchant_avg_amount',
            f'merchant_std_amount', 'merchant_fraud_rate', 'merchant_fraud_count',
            'merchant_unique_users'
        ]

        # Handle NaN in std
        merchant_features['merchant_std_amount'] = merchant_features['merchant_std_amount'].fillna(0)

        return merchant_features

    def aggregate_device_features(self, df):
        """Create device-level aggregated features"""
        df = df.copy()

        # Device-level aggregations
        device_features = df.groupby('device_id').agg({
            'amount': ['count', 'sum', 'mean'],
            'is_fraud': ['mean', 'sum'],
            'user_id': 'nunique',
            'merchant_id': 'nunique'
        }).reset_index()

        # Flatten column names
        device_features.columns = ['device_id'] + [
            f'device_txn_count', f'device_total_amount', f'device_avg_amount',
            'device_fraud_rate', 'device_fraud_count',
            'device_unique_users', 'device_unique_merchants'
        ]

        return device_features

    def create_velocity_features(self, df, time_window_minutes=60):
        """Create transaction velocity features (transactions per time window)"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by entity and timestamp
        df = df.sort_values(['user_id', 'timestamp'])

        # Calculate time differences for each user
        df['user_time_diff'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 60  # minutes

        # Velocity: transactions per hour (inverse of time diff)
        df['user_velocity_txn_per_hour'] = 60 / (df['user_time_diff'] + 1e-6)  # Add small epsilon to avoid div by zero
        df['user_velocity_txn_per_hour'] = df['user_velocity_txn_per_hour'].replace([np.inf, -np.inf], 0)

        # Same for merchants
        df = df.sort_values(['merchant_id', 'timestamp'])
        df['merchant_time_diff'] = df.groupby('merchant_id')['timestamp'].diff().dt.total_seconds() / 60
        df['merchant_velocity_txn_per_hour'] = 60 / (df['merchant_time_diff'] + 1e-6)
        df['merchant_velocity_txn_per_hour'] = df['merchant_velocity_txn_per_hour'].replace([np.inf, -np.inf], 0)

        # Count transactions in last N minutes for each user
        def count_recent_transactions(group, window_minutes):
            timestamps = group['timestamp'].values
            counts = []
            for i, ts in enumerate(timestamps):
                window_start = ts - timedelta(minutes=window_minutes)
                count = np.sum((timestamps >= window_start) & (timestamps <= ts))
                counts.append(count)
            return counts

        df['user_txn_last_60min'] = df.groupby('user_id').apply(
            lambda x: count_recent_transactions(x, 60)
        ).explode().astype(int).values

        df['user_txn_last_10min'] = df.groupby('user_id').apply(
            lambda x: count_recent_transactions(x, 10)
        ).explode().astype(int).values

        return df

    def create_risk_features(self, df):
        """Create risk-based features"""
        df = df.copy()

        # Amount-based risk
        df['amount_log'] = np.log1p(df['amount'])  # log(1+x) to handle zeros
        df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        df['is_low_amount'] = (df['amount'] < df['amount'].quantile(0.05)).astype(int)

        # Amount relative to user history (will be filled after merge)
        df['amount_vs_user_avg'] = 0  # placeholder
        df['amount_vs_user_median'] = 0  # placeholder

        # Time-based risk
        df['is_night_txn'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)

        return df

    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features"""
        df = df.copy()

        if fit or not self.fitted:
            # Fit encoders
            df['user_id_encoded'] = self.user_encoder.fit_transform(df['user_id'])
            df['merchant_id_encoded'] = self.merchant_encoder.fit_transform(df['merchant_id'])
            df['device_id_encoded'] = self.device_encoder.fit_transform(df['device_id'])
            self.fitted = True
        else:
            # Transform using fitted encoders (handle unseen categories)
            df['user_id_encoded'] = self._safe_transform(self.user_encoder, df['user_id'])
            df['merchant_id_encoded'] = self._safe_transform(self.merchant_encoder, df['merchant_id'])
            df['device_id_encoded'] = self._safe_transform(self.device_encoder, df['device_id'])

        # Time of day encoding (already categorical)
        if 'time_of_day' in df.columns:
            time_of_day_dummies = pd.get_dummies(df['time_of_day'], prefix='tod')
            df = pd.concat([df, time_of_day_dummies], axis=1)

        return df

    def _safe_transform(self, encoder, values):
        """Safely transform values, handling unseen categories"""
        transformed = []
        for val in values:
            try:
                transformed.append(encoder.transform([val])[0])
            except ValueError:
                # Unseen category, assign -1 or most frequent class
                transformed.append(-1)
        return np.array(transformed)

    def create_features(self, df, fit=True):
        """
        Main feature creation pipeline
        """
        print("Starting feature engineering...")

        # Step 1: Temporal features
        print("Extracting temporal features...")
        df = self.extract_temporal_features(df)

        # Step 2: Risk features
        print("Creating risk features...")
        df = self.create_risk_features(df)

        # Step 3: Velocity features
        print("Creating velocity features...")
        df = self.create_velocity_features(df)

        # Step 4: Aggregate features (need to compute these first)
        print("Computing user-level features...")
        user_features = self.aggregate_user_features(df)

        print("Computing merchant-level features...")
        merchant_features = self.aggregate_merchant_features(df)

        print("Computing device-level features...")
        device_features = self.aggregate_device_features(df)

        # Step 5: Merge aggregate features back to transactions
        print("Merging aggregate features...")
        df = df.merge(user_features, on='user_id', how='left')
        df = df.merge(merchant_features, on='merchant_id', how='left')
        df = df.merge(device_features, on='device_id', how='left')

        # Step 6: Create relative amount features (after merging user stats)
        print("Creating relative amount features...")
        df['amount_vs_user_avg'] = df['amount'] / (df['user_avg_amount'] + 1e-6)
        df['amount_vs_user_median'] = df['amount'] / (df['user_avg_amount'] + 1e-6)  # using avg as proxy for median

        # Step 7: Enode categorical features
        print("Encoding categorical features...")
        df = self.encode_categorical_features(df, fit=fit)

        # Step 8: Select and prepare final features
        print("Selecting final features...")
        feature_columns = [col for col in df.columns if col not in [
            'transaction_id', 'timestamp', 'user_id', 'merchant_id',
            'device_id', 'ip_address', 'is_fraud', 'fraud_type', 'time_of_day'
        ]]

        # Ensure all feature columns are numeric
        for col in feature_columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric, fill errors with 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        print(f"Feature engineering complete. Created {len(feature_columns)} features.")
        return df, feature_columns

    def get_feature_importance_names(self):
        """Return interpretable feature names for importance plotting"""
        return [
            # Temporal
            'hour', 'day_of_week', 'day_of_month', 'is_weekend',
            'is_night_txn', 'is_business_hours',

            # Amount features
            'amount', 'amount_log', 'is_high_amount', 'is_low_amount',
            'amount_vs_user_avg', 'amount_vs_user_median',

            # User features
            'user_txn_count', 'user_total_amount', 'user_avg_amount',
            'user_std_amount', 'user_min_amount', 'user_max_amount',
            'user_fraud_rate', 'user_fraud_count',
            'user_recent_txn_count_24h', 'user_recent_total_amount_24h',
            'user_recent_avg_amount_24h', 'user_recent_fraud_rate_24h',

            # Merchant features
            'merchant_txn_count', 'merchant_total_amount', 'merchant_avg_amount',
            'merchant_std_amount', 'merchant_fraud_rate', 'merchant_fraud_count',
            'merchant_unique_users',

            # Device features
            'device_txn_count', 'device_total_amount', 'device_avg_amount',
            'device_fraud_rate', 'device_fraud_count',
            'device_unique_users', 'device_unique_merchants',

            # Velocity features
            'user_velocity_txn_per_hour', 'merchant_velocity_txn_per_hour',
            'user_txn_last_60min', 'user_txn_last_10min',

            # Encoded IDs
            'user_id_encoded', 'merchant_id_encoded', 'device_id_encoded',

            # Time of day dummies (if present)
            'tod_night', 'tod_morning', 'tod_afternoon', 'tod_evening'
        ]
