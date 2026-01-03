"""
Run once to train the model with a dataset so that we can use it later for predictions.
This one is for wine.

"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv('winequality-red.csv', delimiter=';')

# Prepare features and target
X = df.drop('quality', axis=1).values
y = df['quality'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'wine_model.joblib')

# Save feature names for reference
feature_names = df.columns[:-1].tolist()
joblib.dump(feature_names, 'feature_names.joblib')

print("Model trained and saved successfully!")
print(f"Features used: {feature_names}")
print(f"Model saved as: wine_model.joblib")
print(f"Feature names saved as: feature_names.joblib")