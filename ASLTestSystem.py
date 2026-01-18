"""
Run once to train the model with a dataset so that we can use it later for predictions.
This one is for  red wine.

"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv('ASLData.csv', delimiter=',')

# Prepare features and target
X = df.drop('sign', axis=1).values
y = df['sign'].values
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'ASL.joblib')

# Save feature names for reference
feature_names = df.columns[:-1].tolist()
joblib.dump(feature_names, 'ASL_feature.joblib')

print("Model trained and saved successfully!")
print(f"Features used: {feature_names}")
print(f"Model saved as: ASL.joblib")
print(f"Feature names saved as: ASL_feature.joblib")