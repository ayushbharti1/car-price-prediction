import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import datetime


# 1. Load Dataset
data = pd.read_csv("car_data.csv")


# 2. Data Cleaning

data['AskPrice'] = (
    data['AskPrice']
    .astype(str)
    .str.replace('₹', '', regex=False)
    .str.replace(',', '', regex=False)
)
data['AskPrice'] = pd.to_numeric(data['AskPrice'], errors='coerce')

data['kmDriven'] = (
    data['kmDriven']
    .astype(str)
    .str.replace('km', '', regex=False)
    .str.replace(',', '', regex=False)
)
data['kmDriven'] = pd.to_numeric(data['kmDriven'], errors='coerce')

data.drop(columns=['model', 'PostedDate', 'AdditionInfo'],
          inplace=True, errors='ignore')

data.dropna(inplace=True)

# Remove extreme outliers
data = data[data['AskPrice'] < data['AskPrice'].quantile(0.99)]
data = data[data['kmDriven'] < data['kmDriven'].quantile(0.99)]


# 3. One-Hot Encoding

data = pd.get_dummies(
    data,
    columns=['Brand', 'FuelType', 'Transmission', 'Owner'],
    drop_first=True
)


# 4. Features & Target

X = data.drop("AskPrice", axis=1)
y = data["AskPrice"]


# 5. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# 6. Model Training

model = RandomForestRegressor(
    n_estimators=400,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# 7. Evaluation

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print("R2 Score :", round(r2, 4))
print("RMSE     :", round(rmse, 2))
print("MAE      :", round(mae, 2))


# 8. Cross Validation

cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("\nAverage Cross-Validation R2:", round(cv_scores.mean(), 4))


# 9. Save Model

joblib.dump(model, "car_price_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("\nModel and metadata saved successfully.")


# 10. USER INPUT PREDICTION (INTERACTIVE)


print("\n===== CAR PRICE PREDICTION =====")

# Load saved model and columns

model = joblib.load("car_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Take input from user
brand = input("Enter Brand (exact dataset name): ")
year = int(input("Enter Manufacturing Year (e.g., 2018): "))
km = int(input("Enter Kilometers Driven: "))
fuel = input("Enter Fuel Type (Petrol/Diesel/etc): ")
transmission = input("Enter Transmission (Manual/Automatic): ")
owner = input("Enter Owner Type (First/Second/etc): ")

current_year = datetime.datetime.now().year
age = current_year - year

# Create base input dictionary
user_input = {
    'Year': year,
    'Age': age,
    'kmDriven': km
}

input_df = pd.DataFrame([user_input])

# Add all missing columns as 0
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Set correct dummy columns
for col in model_columns:
    if col == f"Brand_{brand}":
        input_df[col] = 1
    if col == f"FuelType_{fuel}":
        input_df[col] = 1
    if col == f"Transmission_{transmission}":
        input_df[col] = 1
    if col == f"Owner_{owner}":
        input_df[col] = 1

# Ensure correct column order
input_df = input_df[model_columns]

# Predict
prediction = model.predict(input_df)

print(f"\nEstimated Car Price: ₹{int(prediction[0]):,}")
