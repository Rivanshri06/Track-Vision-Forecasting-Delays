import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np

# Load your dataset
df = pd.read_csv('G:/Rivan/Projects/Projects/Train Delay Prediction/train_delay_data.csv')

# Replace any ellipsis with NaN and handle missing values if necessary
df.replace('...', np.nan, inplace=True)
df.dropna(inplace=True)  # Or apply appropriate imputation

# Handle categorical variables using one-hot encoding
categorical_columns = ['Weather Conditions', 'Day of the Week', 'Time of Day', 'Train Type', 'Route Congestion']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Define target and features
X = df.drop('Historical Delay (min)', axis=1)  # Features
y = df['Historical Delay (min)']  # Target variable

# Save the feature columns for later use in Streamlit app
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, 'feature_columns.pkl')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for later use in Streamlit app
joblib.dump(scaler, 'scaler.pkl')

# Check for NaN or infinite values in X_train and X_test
if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
    raise ValueError("Training data contains NaN or infinite values.")
if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
    raise ValueError("Test data contains NaN or infinite values.")

# Instantiate and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest Mean Squared Error: {mse_rf}')
print(f'Random Forest R-squared: {r2_rf}')

# Instantiate and train the XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the XGBoost model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f'XGBoost Mean Squared Error: {mse_xgb}')
print(f'XGBoost R-squared: {r2_xgb}')

# Instantiate and train the LightGBM Regressor
lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_lgb = lgb_model.predict(X_test)

# Evaluate the LightGBM model
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
r2_lgb = r2_score(y_test, y_pred_lgb)

print(f'LightGBM Mean Squared Error: {mse_lgb}')
print(f'LightGBM R-squared: {r2_lgb}')

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7]
}

# Set up GridSearchCV for XGBoost
grid_search = GridSearchCV(estimator=XGBRegressor(random_state=42), param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f'Best Parameters: {best_params}')
print(f'Best Cross-Validation Score: {best_score}')

# Define base models for stacking
estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
    ('lgb', lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
]

# Define stacking model
stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
)

# Train the stacking model
stacking_model.fit(X_train, y_train)

# Predict and evaluate using the stacking model
try:
    y_pred_stack = stacking_model.predict(X_test)
    if np.any(np.isnan(y_pred_stack)) or np.any(np.isinf(y_pred_stack)):
        raise ValueError("Stacking model predictions contain NaN or infinite values.")
    mse_stack = mean_squared_error(y_test, y_pred_stack)
    r2_stack = r2_score(y_test, y_pred_stack)
    print(f'Stacking Model Mean Squared Error: {mse_stack}')
    print(f'Stacking Model R-squared: {r2_stack}')
except Exception as e:
    print(f'Error in Stacking Model Prediction: {e}')

# Save the best model
joblib.dump(stacking_model, 'best_train_delay_model.pkl')

# To load the saved model later
loaded_model = joblib.load('best_train_delay_model.pkl')
