import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# List of all CSV files
csv_files = [
    '1_1.csv', '2_1.csv', '3_1.csv', '4_1.csv', '5_1.csv',
    'A_1.csv', 'A_2.csv', 'B_1.csv', 'C_1.csv', 'C_2.csv',
    'D_1.csv', 'E_1.csv', 'E_2.csv', 'E_3.csv', 'F_1.csv',
    'I_1.csv', 'L_1.csv', 'O_1.csv', 'S_1.csv', 'S_2.csv',
    'S_3.csv', 'S_4.csv'
]

# Load and combine all CSV files
print("Loading CSV files...")
dataframes = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        dataframes.append(df)
        print(f"  Loaded {file}: {len(df)} rows")
    except FileNotFoundError:
        print(f"  Warning: {file} not found, skipping...")

# Combine all data
df = pd.concat(dataframes, ignore_index=True)
print(f"\nTotal data: {len(df)} rows")

# Display basic info
print(f"Columns: {df.columns.tolist()}")
print(f"Unique Sign values: {df['Sign'].unique()}")
print(f"Sign distribution:\n{df['Sign'].value_counts().sort_index()}")

# Prepare features and target
# Using only Flex sensors as features (since AccelX, GyroX, etc. are all zeros)
feature_columns = ['Flex1', 'Flex2', 'Flex3', 'Flex4', 'Flex5']
X = df[feature_columns].values
y = df['Sign'].values

# Encode the target labels (in case they're strings like 'A', 'B', etc.)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nFeature shape: {X.shape}")
print(f"Target classes: {label_encoder.classes_}")
print(f"Encoded classes: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale the features (optional but recommended for XGBoost, though not strictly necessary)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
print("\n" + "="*60)
print("Training XGBoost Model...")
print("="*60)

# Calculate class weights for imbalanced data
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_train)
weight_dict = dict(zip(np.unique(y_encoded), class_weights))

# Create XGBoost model with optimized parameters
xgb_model = xgb.XGBClassifier(
    n_estimators=200,           # Number of trees
    max_depth=6,                # Maximum tree depth
    learning_rate=0.1,          # Step size shrinkage
    subsample=0.8,              # Subsample ratio of training instances
    colsample_bytree=0.8,       # Subsample ratio of columns
    min_child_weight=1,         # Minimum sum of instance weight in child
    gamma=0,                    # Minimum loss reduction for split
    reg_alpha=0.1,              # L1 regularization
    reg_lambda=1,               # L2 regularization
    random_state=42,
    n_jobs=-1,                  # Use all CPU cores
    eval_metric='mlogloss',     # Multi-class log loss
    use_label_encoder=False
)

# Train the model
xgb_model.fit(
    X_train_scaled, y_train,
    # sample_weight can be added if you want to weight classes
    # sample_weight=[weight_dict[y] for y in y_train],
    verbose=False
)

# Make predictions
y_pred = xgb_model.predict(X_test_scaled)
y_pred_proba = xgb_model.predict_proba(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"\nModel Performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  F1-Score (macro): {f1_macro:.4f}")
print(f"  F1-Score (weighted): {f1_weighted:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[str(c) for c in label_encoder.classes_]))

# Cross-validation
print("\n" + "="*60)
print("Cross-Validation (5-fold):")
print("="*60)
cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
print(f"Individual folds: {cv_scores}")

# Feature importance
print("\n" + "="*60)
print("Feature Importance:")
print("="*60)
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance)

# Confusion Matrix
print("\n" + "="*60)
print("Confusion Matrix:")
print("="*60)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# 2. Feature Importance Bar Plot
axes[1].barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
axes[1].set_xlabel('Importance')
axes[1].set_title('Feature Importance (XGBoost)')
axes[1].invert_yaxis()

# 3. Training History (if available)
results = xgb_model.evals_result() if hasattr(xgb_model, 'evals_result') else None
if results and 'validation_0' in results:
    epochs = len(results['validation_0']['mlogloss'])
    axes[2].plot(range(epochs), results['validation_0']['mlogloss'], label='Train')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Log Loss')
    axes[2].set_title('Training Loss')
    axes[2].legend()
else:
    # Alternative: Show sample distribution
    sign_counts = df['Sign'].value_counts()
    axes[2].bar(range(len(sign_counts)), sign_counts.values, color='coral')
    axes[2].set_xticks(range(len(sign_counts)))
    axes[2].set_xticklabels(sign_counts.index)
    axes[2].set_xlabel('Sign Class')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Class Distribution')

plt.tight_layout()
plt.savefig('xgboost_model_analysis.png', dpi=150)
plt.show()

# Additional analysis: Per-class metrics
print("\n" + "="*60)
print("Per-Class Performance:")
print("="*60)
from sklearn.metrics import precision_recall_fscore_support
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)

for i, class_name in enumerate(label_encoder.classes_):
    print(f"Class {class_name}:")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall: {recall[i]:.4f}")
    print(f"  F1-Score: {fscore[i]:.4f}")
    print(f"  Support: {support[i]}")

# Save the model and preprocessors
import joblib
model_artifacts = {
    'model': xgb_model,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'feature_columns': feature_columns,
    'accuracy': accuracy,
    'f1_score': f1_weighted
}
joblib.dump(model_artifacts, 'xgboost_gesture_model.pkl')
print("\n" + "="*60)
print(f"Model saved to 'xgboost_gesture_model.pkl'")

# Function to predict new samples
def predict_gesture(flex1, flex2, flex3, flex4, flex5, model_artifacts):
    """
    Predict gesture class from 5 flex sensor readings
    """
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    label_encoder = model_artifacts['label_encoder']
    features = np.array([[flex1, flex2, flex3, flex4, flex5]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    return label_encoder.inverse_transform([prediction])[0], probabilities

# Example prediction
print("\n" + "="*60)
print("Example Prediction (using first test sample):")
print("="*60)
sample_flex = X_test[0]
predicted_class, probs = predict_gesture(sample_flex[0], sample_flex[1], sample_flex[2], 
                                          sample_flex[3], sample_flex[4], model_artifacts)
print(f"Flex values: {sample_flex}")
print(f"Predicted gesture: {predicted_class}")
print(f"True gesture: {label_encoder.inverse_transform([y_test[0]])[0]}")
print(f"Probabilities: {dict(zip(label_encoder.classes_, probs))}")

# Hyperparameter tuning suggestion
print("\n" + "="*60)
print("For better performance, try hyperparameter tuning:")
print("="*60)
print("""
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

grid_search = GridSearchCV(xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'), 
                           param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
""")