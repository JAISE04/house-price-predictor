import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import os

def main():
    # Load training data
    df = pd.read_csv("data/train.csv")

    # Split features and target
    X = df.drop(columns=["Id", "SalePrice"])
    y = df["SalePrice"]

    # One-hot encode and handle missing values
    X = pd.get_dummies(X).fillna(0)

    # Save column order for test prediction alignment
    os.makedirs("models", exist_ok=True)
    joblib.dump(X.columns, "models/feature_columns.pkl")

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict and evaluate
    preds = model.predict(X_val_scaled)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"✅ Validation RMSE: {rmse:.2f}")

    # Save model and scaler
    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

if __name__ == "__main__":
    main()
