import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('best_train_delay_model.pkl')

# Load the scaler and feature columns
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

st.title("Train Delay Prediction System")

# Input features
distance = st.number_input("Distance between stations (km)", min_value=0.0, step=1.0)
weather = st.selectbox("Weather Conditions", ["Clear", "Rainy", "Foggy"])
day_of_week = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
train_type = st.selectbox("Train Type", ["Express", "Local", "Regional"])
route_congestion = st.selectbox("Route Congestion", ["Low", "Medium", "High"])

# Function to preprocess input features
def preprocess_input(distance, weather, day_of_week, time_of_day, train_type, route_congestion):
    # Create a dictionary for the input
    input_dict = {
        'distance': distance,
        'Weather Conditions_Rainy': 0,
        'Weather Conditions_Foggy': 0,
        'Day of the Week_Tuesday': 0,
        'Day of the Week_Wednesday': 0,
        'Day of the Week_Thursday': 0,
        'Day of the Week_Friday': 0,
        'Day of the Week_Saturday': 0,
        'Day of the Week_Sunday': 0,
        'Time of Day_Afternoon': 0,
        'Time of Day_Evening': 0,
        'Time of Day_Night': 0,
        'Train Type_Local': 0,
        'Train Type_Regional': 0,
        'Route Congestion_Medium': 0,
        'Route Congestion_High': 0
    }
    
    # Update the dictionary based on user input
    # Weather Conditions
    if weather == "Rainy":
        input_dict['Weather Conditions_Rainy'] = 1
    elif weather == "Foggy":
        input_dict['Weather Conditions_Foggy'] = 1
    
    # Day of the Week
    if day_of_week == "Tuesday":
        input_dict['Day of the Week_Tuesday'] = 1
    elif day_of_week == "Wednesday":
        input_dict['Day of the Week_Wednesday'] = 1
    elif day_of_week == "Thursday":
        input_dict['Day of the Week_Thursday'] = 1
    elif day_of_week == "Friday":
        input_dict['Day of the Week_Friday'] = 1
    elif day_of_week == "Saturday":
        input_dict['Day of the Week_Saturday'] = 1
    elif day_of_week == "Sunday":
        input_dict['Day of the Week_Sunday'] = 1

    # Time of Day
    if time_of_day == "Afternoon":
        input_dict['Time of Day_Afternoon'] = 1
    elif time_of_day == "Evening":
        input_dict['Time of Day_Evening'] = 1
    elif time_of_day == "Night":
        input_dict['Time of Day_Night'] = 1

    # Train Type
    if train_type == "Local":
        input_dict['Train Type_Local'] = 1
    elif train_type == "Regional":
        input_dict['Train Type_Regional'] = 1

    # Route Congestion
    if route_congestion == "Medium":
        input_dict['Route Congestion_Medium'] = 1
    elif route_congestion == "High":
        input_dict['Route Congestion_High'] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing columns with default value 0

    # Reorder columns to match training
    input_df = input_df[feature_columns]

    # Scale the input features
    input_scaled = scaler.transform(input_df)

    return input_scaled

# Make prediction when button is clicked
if st.button('Predict Delay'):
    try:
        # Preprocess the inputs
        features = preprocess_input(distance, weather, day_of_week, time_of_day, train_type, route_congestion)
        
        # Ensure the features are a 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Predict using the loaded model
        prediction = model.predict(features)
        
        st.success(f'Predicted Train Delay: {prediction[0]:.2f} minutes')
    except Exception as e:
        st.error(f'Error in prediction: {e}')
