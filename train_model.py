import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

# Load dataset
df = pd.read_csv("ETO.csv")

# Extract useful time features from 'date'
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

# One-hot encode categorical columns that exist
df_encoded = pd.get_dummies(df, columns=["day_of_week", "holiday", "weather", "route_id"], drop_first=True)

# Separate features and targets
X = df_encoded.drop(["trips", "passengers", "date"], axis=1)
y_trips = df_encoded["trips"]
y_passengers = df_encoded["passengers"]

# Chronological split based on year
train_mask = df["year"] < df["year"].max()
X_train, X_test = X[train_mask], X[~train_mask]
y_trips_train, y_trips_test = y_trips[train_mask], y_trips[~train_mask]
y_pass_train, y_pass_test = y_passengers[train_mask], y_passengers[~train_mask]

# Train Random Forest models
trips_model = RandomForestRegressor(n_estimators=100, random_state=42)
trips_model.fit(X_train, y_trips_train)

passengers_model = RandomForestRegressor(n_estimators=100, random_state=42)
passengers_model.fit(X_train, y_pass_train)

# Evaluate Trips model
y_trips_pred = trips_model.predict(X_test)
mae_trips = mean_absolute_error(y_trips_test, y_trips_pred)
rmse_trips = mean_squared_error(y_trips_test, y_trips_pred, squared=False)
r2_trips = r2_score(y_trips_test, y_trips_pred)

print("\n Trips Model Evaluation (Random Forest, Chronological Split):")
print(f"MAE  : {mae_trips:.2f}")
print(f"RMSE : {rmse_trips:.2f}")
print(f"R²   : {r2_trips:.4f}")

# Evaluate Passengers model
y_pass_pred = passengers_model.predict(X_test)
mae_pass = mean_absolute_error(y_pass_test, y_pass_pred)
rmse_pass = mean_squared_error(y_pass_test, y_pass_pred, squared=False)
r2_pass = r2_score(y_pass_test, y_pass_pred)

print("\n Passengers Model Evaluation (Random Forest, Chronological Split):")
print(f"MAE  : {mae_pass:.2f}")
print(f"RMSE : {rmse_pass:.2f}")
print(f"R²   : {r2_pass:.4f}")

# Save trained models
joblib.dump(trips_model, "trips_model.pkl")
joblib.dump(passengers_model, "passengers_model.pkl")

print("\n✅ Random Forest Models trained and saved successfully")
