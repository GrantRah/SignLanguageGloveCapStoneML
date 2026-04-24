import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
import joblib

# =========================
# REMOVE WARNINGS / NOISE
# =========================
warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
DATA_FOLDER = "."
MODEL_PATH = "asl_flex_model.pkl"

# =========================
# LOAD CSV FILES
# =========================
csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]

if len(csv_files) == 0:
    raise ValueError("No CSV files found!")

dataframes = []

for file in csv_files:
    df = pd.read_csv(os.path.join(DATA_FOLDER, file))

    if "Sign" not in df.columns:
        print(f"Skipping {file} (no Sign column)")
        continue

    df['Sign'] = df['Sign'].astype(str)
    dataframes.append(df)
    print(f"Loaded {file}: {len(df)} rows")

df = pd.concat(dataframes, ignore_index=True)

print("\nTotal samples:", len(df))

# =========================
# FLEX ONLY
# =========================
flex_cols = ['Flex1', 'Flex2', 'Flex3', 'Flex4', 'Flex5']

# =========================
# WINDOW FEATURE EXTRACTION
# =========================
WINDOW_SIZE = 20
STEP_SIZE = 10

rows = []

for file in csv_files:
    df_file = pd.read_csv(os.path.join(DATA_FOLDER, file))

    if "Sign" not in df_file.columns:
        continue

    df_file['Sign'] = df_file['Sign'].astype(str)

    flex_data = df_file[flex_cols].values
    label = df_file['Sign'].iloc[0]

    for i in range(0, len(flex_data) - WINDOW_SIZE, STEP_SIZE):
        window = flex_data[i:i+WINDOW_SIZE]

        avg = np.mean(window, axis=0)
        std = np.std(window, axis=0)
        ratio = avg / (np.sum(avg) + 1e-8)

        features = np.concatenate([avg, std, ratio])
        rows.append(list(features) + [label])

feature_columns = (
    [f"mean_Flex{i}" for i in range(1,6)] +
    [f"std_Flex{i}" for i in range(1,6)] +
    [f"ratio_Flex{i}" for i in range(1,6)]
)

processed_df = pd.DataFrame(rows, columns=feature_columns + ['Sign'])

print("\nAfter windowing:", processed_df.shape)

# =========================
# SPLIT FEATURES / LABELS
# =========================
X = processed_df[feature_columns].values
y = processed_df['Sign'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =========================
# SCALE FEATURES
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# XGBOOST BASE MODEL (quiet)
# =========================
xgb = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    verbosity=0,
    random_state=42,
    n_jobs=-1
)

# =========================
# RANDOMIZED SEARCH CV
# =========================
param_dist = {
    "n_estimators": [200, 400, 600],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "gamma": [0, 0.1, 0.2],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=25,
    cv=cv,
    scoring='accuracy',
    verbose=0,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train_scaled, y_train)

print("\nBest Parameters:")
print(search.best_params_)

best_model = search.best_estimator_

# =========================
# CROSS VALIDATION SCORE (FINAL CHECK)
# =========================
cv_scores = cross_val_score(
    best_model,
    X_train_scaled,
    y_train,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

print("\nCV Mean Accuracy:", cv_scores.mean())
print("CV Std:", cv_scores.std())

# =========================
# CALIBRATION
# =========================
model = CalibratedClassifierCV(best_model, method='sigmoid', cv=5)
model.fit(X_train_scaled, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n========================")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("========================")

# =========================
# SAMPLE PREDICTIONS
# =========================
print("\nSample predictions:")
for i in range(min(10, len(y_pred))):
    pred_label = label_encoder.inverse_transform([y_pred[i]])[0]
    true_label = label_encoder.inverse_transform([y_test[i]])[0]
    conf = np.max(y_proba[i])

    print(f"True={true_label} | Pred={pred_label} | Conf={conf:.3f}")

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_, rotation=45)
plt.yticks(range(len(label_encoder.classes_)), label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# =========================
# FEATURE IMPORTANCE
# =========================
importances = best_model.feature_importances_

plt.figure(figsize=(8,6))
plt.barh(feature_columns, importances)
plt.title("Feature Importance (Mean + Std + Ratio)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# =========================
# SAVE MODEL
# =========================
artifacts = {
    "model": model,
    "scaler": scaler,
    "label_encoder": label_encoder,
    "features": feature_columns
}

joblib.dump(artifacts, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)
