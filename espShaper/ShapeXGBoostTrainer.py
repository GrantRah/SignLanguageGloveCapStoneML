"""
Trains model on raw Ax, Ay, Az gesture blocks
Includes differentiation (np.diff),  this is using the XGBoost model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import joblib

# =========================
# CONFIG
# =========================
CSV_PATH = r"C:\Users\Grant\OneDrive\Documents\PythonFiles\shape_features.csv"
MODEL_PATH = "Trained_Shapes_XGBoost.joblib"
FEATURE_PATH = "feature_Shapes_XGBoost.joblib"
ENCODER_PATH = "label_encoder.joblib"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

print("Loaded shape:", df.shape)

if 'gesture_id' not in df.columns:
    raise ValueError("CSV must contain 'gesture_id' column!")

# =========================
# SPLIT FEATURES / LABELS
# =========================
X = df.drop(['gesture_id'], axis=1)
y = df['gesture_id']

feature_names = X.columns.tolist()

# =========================
# ENCODE LABELS (REQUIRED FOR XGBOOST)
# =========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X = X.values

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.3,
    random_state=42,
    stratify=y_encoded
)

# =========================
# TRAIN MODEL (TUNED)
# =========================
model = XGBClassifier(
    n_estimators=400,          # more trees → better learning
    max_depth=5,               # slightly lower to prevent overfit
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,

    reg_lambda=1.0,            # L2 regularization
    reg_alpha=0.2,             # L1 regularization (helps confidence)

    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"\nTrain Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")

# =========================
# CONFIDENCE + LABEL DECODE
# =========================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print("\nSample predictions with confidence:")
for i in range(min(5, len(y_pred))):
    pred_label = le.inverse_transform([y_pred[i]])[0]
    conf = np.max(y_proba[i])
    print(f"Pred: {pred_label} | Confidence: {conf:.3f}")

# =========================
# CONFUSION MATRIX (FIXED LABELS)
# =========================
cm = confusion_matrix(y_test, y_pred)

labels = le.inverse_transform(np.unique(y_encoded))

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Heatmap)")
plt.show()

# =========================
# FEATURE IMPORTANCE
# =========================
importances = model.feature_importances_
indices = np.argsort(importances)[-15:]

plt.figure(figsize=(8, 6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Importance")
plt.title("Top Feature Importance")
plt.show()

# =========================
# SAVE EVERYTHING
# =========================
joblib.dump(model, MODEL_PATH)
joblib.dump(feature_names, FEATURE_PATH)
joblib.dump(le, ENCODER_PATH)

print("\nSaved model to:", MODEL_PATH)
print("Saved features to:", FEATURE_PATH)
print("Saved label encoder to:", ENCODER_PATH)

# =========================
# DEBUG INFO
# =========================
print("\nClass distribution:")
print(df['gesture_id'].value_counts())