import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Plot churn distribution
sns.countplot(x='Churn', data=data)
plt.savefig('churn_distribution.png')

# Plot tenure vs. churn
sns.boxplot(x='Churn', y='tenure', data=data)
plt.savefig('tenure_vs_churn.png')

# Correlation heatmap
numerical_data = data.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numerical_data.corr(), annot=True)
plt.savefig('correlation_heatmap.png')

