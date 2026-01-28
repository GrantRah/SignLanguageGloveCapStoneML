"""
Run once to train the model with a dataset so that we can use it later for predictions.
This one is for Nano.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv(
    r"C:\Users\Grant\OneDrive\Documents\PythonFiles\zGlove.py\gesture_dataahh.csv",
    delimiter=','
)

# === SANITY CHECKS ===
# Remove any extra whitespace from column names
df.columns = df.columns.str.strip()

# Print loaded columns to verify CSV headers
print("Columns loaded:", df.columns.tolist())

# Print data shape to verify correct number of rows and columns
print("Data shape:", df.shape)

# Print first few rows to check that data looks correct
print("First few rows:\n", df.head())

# Ensure 'Gesture' column exists to prevent KeyError
if 'Gesture' not in df.columns:
    raise ValueError("The CSV does not contain a 'Gesture' column!")
# ====================

# Prepare features and target
X = df.drop('Gesture', axis=1).values
y = df['Gesture'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'Trained_Nano_model.joblib')

# Save feature names for reference
feature_names = df.columns[:-1].tolist()
joblib.dump(feature_names, 'feature_Nano.joblib')

print("Model trained and saved successfully!")
print(f"Features used: {feature_names}")
print(f"Model saved as: Trained_Nano_model.joblib")
print(f"Feature names saved as: feature_Nano.joblib")
