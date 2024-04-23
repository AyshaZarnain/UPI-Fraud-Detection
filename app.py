from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Load the trained model
model = DecisionTreeClassifier()
# Assuming you have trained the model previously

app = Flask(__name__, template_folder='C:\\Users\\aysha\\Downloads\\Fraud\\OnlineFraud\\templates')
@app.route('/')
def home():
    return render_template('OnlineFraud/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    transaction_type = int(data['type'])
    amount = float(data['amount'])
    old_balance = float(data['oldbalanceOrg'])
    new_balance = float(data['newbalanceOrig'])

    features = np.array([[transaction_type, amount, old_balance, new_balance]])
    prediction = model.predict(features)

    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
