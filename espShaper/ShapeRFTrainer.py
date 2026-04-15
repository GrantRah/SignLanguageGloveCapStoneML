"""
Train RandomForest on raw Ax, Ay, Az gesture blocks
Includes differentiation (np.diff) with the random forest model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

# =========================
# CONFIG
# =========================
CSV_PATH = r"C:\Users\Grant\OneDrive\Documents\PythonFiles\shape_features.csv"
MODEL_PATH = "Trained_Shapes_model.joblib"
FEATURE_PATH = "feature_Shapes.joblib"

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

X = X.values
y = y.values

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =========================
# TRAIN MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
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
# CONFIDENCE EXAMPLE (IMPORTANT)
# =========================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print("\nSample predictions with confidence:")
for i in range(min(5, len(y_pred))):
    conf = np.max(y_proba[i])
    print(f"Pred: {y_pred[i]} | Confidence: {conf:.3f}")

# =========================
# CONFUSION MATRIX HEATMAP
# =========================
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Heatmap)")
plt.show()

# =========================
# FEATURE IMPORTANCE
# =========================
importances = model.feature_importances_
indices = np.argsort(importances)[-15:]  # top 15 features

plt.figure(figsize=(8, 6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Importance")
plt.title("Top Feature Importance")
plt.show()

# =========================
# SAVE MODEL + FEATURES
# =========================
joblib.dump(model, MODEL_PATH)
joblib.dump(feature_names, FEATURE_PATH)

print("\nSaved model to:", MODEL_PATH)
print("Saved features to:", FEATURE_PATH)

# =========================
# CLASS DISTRIBUTION DEBUG
# =========================
print("\nClass distribution:")
print(df['gesture_id'].value_counts())