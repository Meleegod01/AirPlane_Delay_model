from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.utils import resample

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("airline_delay.csv")

# Handling missing values
data = data.fillna(0)

# Handle date column
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['month'] = data['date'].dt.month

# Create the 'is_holiday' feature
holiday_list = [{'month': 1, 'year': 2020}, {'month': 12, 'year': 2020}]
data['is_holiday'] = data.apply(lambda row: 1 if {'month': row['month'], 'year': row['year']} in holiday_list else 0, axis=1)

# Define features and target
features = ['arr_flights', 'arr_del15', 'carrier_ct', 'weather_ct', 'security_ct', 'late_aircraft_ct', 'is_holiday']
target = 'arr_delay'

# Prepare data for training
X = data[features].fillna(0)
y = data[target].fillna(0)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

def calculate_confidence(model, X, num_iterations=100):
    predictions = []
    for _ in range(num_iterations):
        # Resample the data with replacement
        X_resampled = resample(X, replace=True)
        # Predict using the model
        preds = model.predict(X_resampled)
        predictions.append(preds)
    
    # Compute mean and std deviation of predictions
    predictions = np.array(predictions)
    mean_predictions = np.mean(predictions, axis=0)
    std_predictions = np.std(predictions, axis=0)
    
    return mean_predictions, std_predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        arr_flights = float(request.form['arr_flights'])
        arr_del15 = float(request.form['arr_del15'])
        carrier_ct = float(request.form['carrier_ct'])
        weather_ct = float(request.form['weather_ct'])
        security_ct = float(request.form['security_ct'])
        late_aircraft_ct = float(request.form['late_aircraft_ct'])
        is_holiday = int(request.form['is_holiday'])

        # Feature array
        input_features = np.array([[arr_flights, arr_del15, carrier_ct, weather_ct, security_ct, late_aircraft_ct, is_holiday]])

        # Standardize the input
        input_features_scaled = scaler.transform(input_features)

        # Make prediction
        prediction = model.predict(input_features_scaled)
        prediction = round(prediction[0], 2)

        # Calculate confidence
        mean_pred, std_pred = calculate_confidence(model, input_features_scaled)
        confidence_percentage = np.mean((mean_pred - 1.96 * std_pred <= prediction) & 
                                        (prediction <= mean_pred + 1.96 * std_pred)) * 100

        return render_template('index.html', 
                               prediction_text=f'Predicted Arrival Delay: {prediction} minutes',
                               confidence_percentage=round(confidence_percentage, 2),
                               arr_flights=arr_flights, arr_del15=arr_del15, carrier_ct=carrier_ct,
                               weather_ct=weather_ct, security_ct=security_ct, late_aircraft_ct=late_aircraft_ct,
                               is_holiday=is_holiday)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)