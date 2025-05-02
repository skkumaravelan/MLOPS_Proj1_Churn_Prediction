import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def monitor_predictions(log_file='predictions.csv'):
    try:
        # Load predictions
        df = pd.read_csv(log_file)
        if df.empty:
            print("No predictions logged yet.")
            return

        # Basic stats
        print("Prediction Summary:")
        print(f"Total Predictions: {len(df)}")
        print(f"Churn Rate: {df['prediction'].mean():.2%}")
        print(f"Average Probability: {df['probability'].mean():.2f}")

        # Check for drift in key features (e.g., tenure, MonthlyCharges)
        print("\\nFeature Statistics:")
        for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            mean = df[col].mean()
            std = df[col].std()
            print(f"{col} - Mean: {mean:.2f}, Std: {std:.2f}")

        # Plot prediction distribution
        plt.figure(figsize=(8, 6))
        df['prediction'].value_counts().plot(kind='bar', title='Churn Prediction Distribution')
        plt.xlabel('Churn (0 = No, 1 = Yes)')
        plt.ylabel('Count')
        plt.savefig('churn_distribution.png')
        print("Saved churn distribution plot as 'churn_distribution.png'")

    except Exception as e:
        print(f"Error in monitoring: {str(e)}")

if __name__ == '__main__':
    monitor_predictions()

