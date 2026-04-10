from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)
# Load model files
current_dir = os.path.dirname(os.path.abspath(__file__))

model = None
scaler = None
label_encoders = None

try:
    model = joblib.load(os.path.join(current_dir, 'model.pkl'))
    scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))
    label_encoders = joblib.load(os.path.join(current_dir, 'label_encoders.pkl'))
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)

# ❗ Removed Loan_ID
FEATURE_NAMES = [
    'Gender', 'Married', 'Dependents', 'Education',
    'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
    'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('result.html',
                               prediction="Model not loaded",
                               error=True)

    try:
        # Get form data
        input_data = {
            'Gender': request.form['gender'],
            'Married': request.form['married'],
            'Dependents': request.form['dependents'],
            'Education': request.form['education'],
            'Self_Employed': request.form['self_employed'],
            'ApplicantIncome': float(request.form['applicant_income']),
            'CoapplicantIncome': float(request.form['coapplicant_income']),
            'LoanAmount': float(request.form['loan_amount']),
            'Loan_Amount_Term': float(request.form['loan_term']),
            'Credit_History': float(request.form['credit_history']),
            'Property_Area': request.form['property_area']
        }

        # Validation
        if input_data['ApplicantIncome'] <= 0:
            raise ValueError("Income must be greater than 0")

        if input_data['LoanAmount'] <= 0:
            raise ValueError("Loan amount must be greater than 0")

        # Encode categorical values
        for col in FEATURE_NAMES:
            if col in label_encoders:
                value = input_data[col]

                if col == 'Dependents':
                    value = '3+' if value == '3+' else str(int(value))

                if value in label_encoders[col].classes_:
                    input_data[col] = label_encoders[col].transform([value])[0]
                else:
                    input_data[col] = 0

        # Create feature vector
        features = np.array([float(input_data[f]) for f in FEATURE_NAMES]).reshape(1, -1)

        # Scale
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        if prediction == 1:
            result = "APPROVED"
            confidence = probability[1] * 100
        else:
            result = "REJECTED"
            confidence = probability[0] * 100

        return render_template('result.html',
                               prediction=result,
                               probability=round(confidence, 2),
                               error=False,
                               applicant_income=input_data['ApplicantIncome'],
                               loan_amount=input_data['LoanAmount'],
                               credit_history=input_data['Credit_History'])

    except Exception as e:
        return render_template('result.html',
                               prediction=str(e),
                               error=True)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()

        input_data = {
            'Gender': data['Gender'],
            'Married': data['Married'],
            'Dependents': str(data.get('Dependents', '0')),
            'Education': data.get('Education', 'Graduate'),
            'Self_Employed': data.get('Self_Employed', 'No'),
            'ApplicantIncome': float(data['ApplicantIncome']),
            'CoapplicantIncome': float(data.get('CoapplicantIncome', 0)),
            'LoanAmount': float(data['LoanAmount']),
            'Loan_Amount_Term': float(data.get('Loan_Amount_Term', 360)),
            'Credit_History': float(data.get('Credit_History', 1)),
            'Property_Area': data.get('Property_Area', 'Urban')
        }

        # Encoding
        for col in FEATURE_NAMES:
            if col in label_encoders:
                val = input_data[col]
                if val in label_encoders[col].classes_:
                    input_data[col] = label_encoders[col].transform([val])[0]
                else:
                    input_data[col] = 0

        features = np.array([float(input_data[f]) for f in FEATURE_NAMES]).reshape(1, -1)
        features_scaled = scaler.transform(features)

        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0]

        return jsonify({
            "prediction": "Approved" if pred == 1 else "Rejected",
            "confidence": round(max(prob) * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
