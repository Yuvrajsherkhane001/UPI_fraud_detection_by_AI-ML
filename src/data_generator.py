# src/data_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_day1_transactions(n_samples=10000):
    """Generate Day 1 dataset with clear fraud patterns"""
    np.random.seed(42)
    data = []

    for i in range(n_samples):
        # Normal transaction (70%)
        if random.random() < 0.7:
            txn = {
                'transaction_id': f'TXN_{i:06d}',
                'timestamp': datetime(2026, 9, 8) + timedelta(
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59)
                ),
                'user_id': f'USER_{random.randint(1000, 5000):04d}',
                'merchant_id': f'MERCH_{random.randint(100, 500):03d}',
                'amount': round(random.uniform(10, 5000), 2),
                'device_id': f'DEV_{random.randint(100, 300):03d}',
                'ip_address': f'{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}',
                'is_fraud': 0,
                'fraud_type': 'none'
            }
        # Cardholder fraud (20%) - same card used rapidly at multiple merchants
        elif random.random() < 0.9:
            base_time = datetime(2026, 9, 8) + timedelta(
                hours=random.randint(0, 20),
                minutes=random.randint(0, 50)
            )
            txn = {
                'transaction_id': f'TXN_{i:06d}',
                'timestamp': base_time,
                'user_id': f'USER_{random.randint(1000, 5000):04d}',
                'merchant_id': f'MERCH_{random.randint(100, 500):03d}',
                'amount': round(random.uniform(5000, 50000), 2),
                'device_id': f'DEV_{random.randint(100, 300):03d}',
                'ip_address': f'{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}',
                'is_fraud': 1,
                'fraud_type': 'cardholder'
            }
        # Merchant fraud (10%) - new merchant + high amount + same-day payout
        else:
            txn = {
                'transaction_id': f'TXN_{i:06d}',
                'timestamp': datetime(2026, 9, 8) + timedelta(
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59)
                ),
                'user_id': f'USER_{random.randint(1000, 5000):04d}',
                'merchant_id': f'NEW_MERCH_{random.randint(1000, 2000):04d}',  # New merchant
                'amount': round(random.uniform(10000, 100000), 2),  # High amount
                'device_id': f'DEV_{random.randint(100, 300):03d}',
                'ip_address': f'{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}',
                'is_fraud': 1,
                'fraud_type': 'merchant'
            }
        data.append(txn)

    return pd.DataFrame(data)

def generate_day2_transactions(n_samples=10000):
    """Generate Day 2 dataset with evolved/fraud patterns"""
    np.random.seed(43)  # Different seed for variation
    data = []

    for i in range(n_samples):
        # Normal transaction (65% - slightly less than Day 1 as fraud evolves)
        if random.random() < 0.65:
            txn = {
                'transaction_id': f'TXN_{i:06d}',
                'timestamp': datetime(2026, 9, 9) + timedelta(
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59)
                ),
                'user_id': f'USER_{random.randint(1000, 5000):04d}',
                'merchant_id': f'MERCH_{random.randint(100, 500):03d}',
                'amount': round(random.uniform(10, 5000), 2),
                'device_id': f'DEV_{random.randint(100, 300):03d}',
                'ip_address': f'{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}',
                'is_fraud': 0,
                'fraud_type': 'none'
            }
        # Evolved cardholder fraud (25%) - split transactions, 2-day spread
        elif random.random() < 0.9:
            base_time = datetime(2026, 9, 9) + timedelta(
                hours=random.randint(0, 20),
                minutes=random.randint(0, 50)
            )
            # Sometimes spread across 2 days
            if random.random() < 0.3:
                base_time -= timedelta(days=1)

            txn = {
                'transaction_id': f'TXN_{i:06d}',
                'timestamp': base_time,
                'user_id': f'USER_{random.randint(1000, 5000):04d}',
                'merchant_id': f'MERCH_{random.randint(100, 500):03d}',
                'amount': round(random.uniform(1000, 10000), 2),  # Smaller, split amounts
                'device_id': f'DEV_{random.randint(100, 300):03d}',
                'ip_address': f'{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}',
                'is_fraud': 1,
                'fraud_type': 'cardholder_evolved'
            }
        # Evolved merchant fraud (10%) - same-category merchants, 2-hour timespan
        else:
            # Create a timespan of transactions within 2 hours
            base_time = datetime(2026, 9, 9) + timedelta(
                hours=random.randint(0, 22),
                minutes=random.randint(0, 50)
            )

            txn = {
                'transaction_id': f'TXN_{i:06d}',
                'timestamp': base_time + timedelta(minutes=random.randint(0, 120)),  # Within 2-hour window
                'user_id': f'USER_{random.randint(1000, 5000):04d}',
                'merchant_id': f'MERCH_{random.randint(100, 500):03d}_{random.choice(["FOOD", "RETAIL", "TRAVEL", "ENTERTAIN"])}',  # Same category
                'amount': round(random.uniform(5000, 30000), 2),
                'device_id': f'DEV_{random.randint(100, 300):03d}',
                'ip_address': f'{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}',
                'is_fraud': 1,
                'fraud_type': 'merchant_evolved'
            }
        data.append(txn)

    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate and save datasets
    print("Generating Day 1 transactions...")
    day1_df = generate_day1_transactions(10000)
    day1_df.to_csv('data/day1_transactions.csv', index=False)
    print(f"Day 1: {len(day1_df)} transactions ({day1_df['is_fraud'].sum()} fraudulent)")

    print("Generating Day 2 transactions...")
    day2_df = generate_day2_transactions(10000)
    day2_df.to_csv('data/day2_transactions.csv', index=False)
    print(f"Day 2: {len(day2_df)} transactions ({day2_df['is_fraud'].sum()} fraudulent)")

    print("\nDataset generation complete! Files saved in data/ directory")
