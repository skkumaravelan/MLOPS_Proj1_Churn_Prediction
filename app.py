from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
from datetime import datetime

app = Flask(__name__)

# Load model
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize predictions CSV
LOG_FILE = 'predictions.csv'
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=['timestamp', 'prediction', 'probability'] +
                 ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male',
                  'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service',
                  'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No',
                  'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
                  'OnlineBackup_No internet service', 'OnlineBackup_Yes',
                  'DeviceProtection_No internet service', 'DeviceProtection_Yes',
                  'TechSupport_No internet service', 'TechSupport_Yes',
                  'StreamingTV_No internet service', 'StreamingTV_Yes',
                  'StreamingMovies_No internet service', 'StreamingMovies_Yes',
                  'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
                  'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
                  'PaymentMethod_Mailed check']).to_csv(LOG_FILE, index=False)

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data as JSON
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Predict churn directly
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Log prediction
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'probability': probability,
            **data
        }
        pd.DataFrame([log_entry]).to_csv(LOG_FILE, mode='a', header=False, index=False)

        # Return result
        return jsonify({
            'churn': int(prediction),
            'probability': float(probability)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

