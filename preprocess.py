import pandas as pd

# Load dataset
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Handle missing values
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)

# Drop non-numeric columns (e.g., customerID)
data = data.drop(['customerID'], axis=1)

# Identify all categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

# Encode categorical variables (including Churn)
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Save preprocessed data
data.to_csv('processed_data.csv', index=False)
print("Data preprocessing complete.")
