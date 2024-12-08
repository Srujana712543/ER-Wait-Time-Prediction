from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Route to render the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction API route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Extract input features from the JSON request
    arrival_hour = int(data['arrival_hour'])
    arrival_weekday = int(data['arrival_weekday'])
    arrival_month = int(data['arrival_month'])
    time_of_day = int(data['time_of_day'])
    staff_available = int(data['staff_available'])
    patient_acuity = int(data['patient_acuity'])
    
    # Prepare the input for the model
    features = np.array([[arrival_hour, arrival_weekday, arrival_month, time_of_day, staff_available, patient_acuity]])
    
    # Make prediction
    predicted_wait_time = model.predict(features)[0]
    
    # Return the result as JSON
    return jsonify({"wait_time": round(predicted_wait_time, 2)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
