import pandas as pd

# Load dataset
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Handle missing values
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)

# Encode categorical variables
categorical_cols = ['gender', 'Contract', 'PaymentMethod', 'Churn']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Save preprocessed data
data.to_csv('processed_data.csv', index=False)
print("Data preprocessing complete.")
