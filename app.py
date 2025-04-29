from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load model
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data as JSON
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Predict churn directly (no scaling)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Return result
        return jsonify({
            'churn': int(prediction),
            'probability': float(probability)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
