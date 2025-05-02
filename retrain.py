import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

def retrain_model(data_file='processed_data.csv', model_file='churn_model.pkl'):
    try:
        # Load data
        data = pd.read_csv(data_file)
        X = data.drop('Churn_Yes', axis=1)
        y = data['Churn_Yes']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train new model
        new_model = LogisticRegression(max_iter=1000)
        new_model.fit(X_train, y_train)

        # Evaluate
        y_pred = new_model.predict(X_test)
        new_accuracy = accuracy_score(y_test, y_pred)
        print(f"New Model Accuracy: {new_accuracy:.4f}")

        # Compare with existing model (if exists)
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                old_model = pickle.load(f)
            old_pred = old_model.predict(X_test)
            old_accuracy = accuracy_score(y_test, old_pred)
            print(f"Old Model Accuracy: {old_accuracy:.4f}")

            if new_accuracy > old_accuracy:
                with open(model_file, 'wb') as f:
                    pickle.dump(new_model, f)
                print(f"Updated model saved as {model_file}")
            else:
                print("New model not better; keeping old model.")
        else:
            with open(model_file, 'wb') as f:
                pickle.dump(new_model, f)
            print(f"New model saved as {model_file}")

    except Exception as e:
        print(f"Error in retraining: {str(e)}")

if __name__ == '__main__':
    retrain_model()

