import pandas as pd
import joblib

# Load model, scaler, and columns
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/feature_columns.pkl")

# Load test set
test_df = pd.read_csv("data/test.csv")
X_test = pd.get_dummies(test_df).fillna(0)

# Align with training columns
X_test = X_test.reindex(columns=columns, fill_value=0)

# Scale
X_test_scaled = scaler.transform(X_test)

# Predict
preds = model.predict(X_test_scaled)

# Save submission
submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": preds
})
submission.to_csv("outputs/submission.csv", index=False)
print("✅ Submission saved to outputs/submission.csv")
