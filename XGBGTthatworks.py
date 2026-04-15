import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# List of your CSV files (update the path as needed)
# Option 1: If files are in current directory
csv_files = [
    '1_1.csv', '2_1.csv', '3_1.csv', '4_1.csv', '5_1.csv',
    'A_1.csv', 'A_2.csv', 'B_1.csv', 'C_1.csv', 'C_2.csv',
    'D_1.csv', 'E_1.csv', 'E_2.csv', 'E_3.csv', 'F_1.csv',
    'I_1.csv', 'L_1.csv', 'O_1.csv', 'S_1.csv', 'S_2.csv',
    'S_3.csv', 'S_4.csv'
]

# Option 2: If files are in a different location, add the path
# base_path = r'C:\Users\capta\Downloads'  # Change this to your actual path
# csv_files = [os.path.join(base_path, f) for f in csv_files]

# Load the files
dataframes = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        dataframes.append(df)
        print(f"✓ Loaded {file}: {len(df)} rows")
    except FileNotFoundError:
        print(f"✗ File not found: {file}")

if len(dataframes) == 0:
    print("\n❌ No files were loaded!")
    print("\nPlease check:")
    print("1. Are the CSV files in the same directory as this script?")
    print("2. Current directory:", os.getcwd())
    print("3. Files in current directory:", os.listdir())
else:
    # Combine all data
    df = pd.concat(dataframes, ignore_index=True)
    print(f"\n✅ Total data: {len(df)} rows")
    
    # Continue with training...
    feature_columns = ['Flex1', 'Flex2', 'Flex3', 'Flex4', 'Flex5']
    X = df[feature_columns].values
    y = df['Sign'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    print("\nTraining XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    xgb_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = xgb_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in label_encoder.classes_]))
    
    print("\n✅ Training completed successfully!")