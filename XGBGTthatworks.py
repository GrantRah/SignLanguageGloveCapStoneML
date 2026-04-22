import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# List of specific CSV files to load (as provided earlier)
specific_csv_files = [
    '1_1.csv', '2_1.csv', '3_1.csv', '4_1.csv', '5_1.csv',
    'A_1.csv', 'A_2.csv', 'B_1.csv', 'C_1.csv', 'C_2.csv',
    'D_1.csv', 'E_1.csv', 'E_2.csv', 'E_3.csv', 'F_1.csv',
    'I_1.csv', 'L_1.csv', 'O_1.csv', 'S_1.csv', 'S_2.csv',
    'S_3.csv', 'S_4.csv'
]

# Define finger names mapping
finger_names = {
    'Flex1': 'Thumb',
    'Flex2': 'Index',
    'Flex3': 'Middle',
    'Flex4': 'Ring',
    'Flex5': 'Pinky'
}

print(f"Looking for {len(specific_csv_files)} specific CSV files...")
print(f"Files to load: {specific_csv_files[:5]}... (showing first 5)")

# Filter to only include files that actually exist
csv_files = []
for file in specific_csv_files:
    if os.path.exists(file):
        csv_files.append(file)
    else:
        print(f"⚠️ File not found: {file}")

print(f"\nFound {len(csv_files)} out of {len(specific_csv_files)} CSV files")

if len(csv_files) == 0:
    print("\n❌ No CSV files found in current directory!")
    print(f"Current directory: {os.getcwd()}")
    print("\nPlease ensure the CSV files are in the current directory or update the paths.")
else:
    # Load the specific CSV files
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
            print(f"✓ Loaded {file}: {len(df)} rows")
        except Exception as e:
            print(f"✗ Error loading {file}: {e}")
    
    # Combine all data
    df = pd.concat(dataframes, ignore_index=True)
    print(f"\n✅ Total data: {len(df)} rows")
    
    # Convert Sign column to string to handle mixed types
    df['Sign'] = df['Sign'].astype(str)
    
    # Display basic info
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"Unique Sign values: {sorted(df['Sign'].unique())}")
    print(f"\nSign distribution:")
    print(df['Sign'].value_counts().sort_index())
    
    # Prepare features and target
    feature_columns = ['Flex1', 'Flex2', 'Flex3', 'Flex4', 'Flex5']
    X = df[feature_columns].values
    y = df['Sign'].values
    
    # Encode labels (convert string labels to numbers)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Get class names as strings
    class_names = [str(cls) for cls in label_encoder.classes_]
    
    print(f"\nFeature shape: {X.shape}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes mapping:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name} -> {i}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model
    print("\n" + "="*60)
    print("Training XGBoost Model...")
    print("="*60)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    xgb_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = xgb_model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n📊 Model Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score (macro): {f1_macro:.4f}")
    print(f"  F1-Score (weighted): {f1_weighted:.4f}")
    
    # Classification report - using string class names
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Feature importance with finger names
    print("\n" + "="*60)
    print("🎯 Feature Importance (by Finger):")
    print("="*60)
    feature_importance = pd.DataFrame({
        'Sensor': feature_columns,
        'Finger': [finger_names[f] for f in feature_columns],
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(feature_importance.to_string(index=False))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n📊 Confusion Matrix:")
    print(cm)
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names,
                yticklabels=class_names)
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('Actual', fontsize=12)
    
    # Adjust label rotation if needed
    if len(class_names) > 10:
        axes[0].set_xticklabels(class_names, rotation=45, ha='right')
        axes[0].set_yticklabels(class_names, rotation=0)
    
    # 2. Feature Importance Bar Plot with finger names
    # Create a color map for different fingers
    colors = plt.cm.Set3(np.linspace(0, 1, len(feature_importance)))
    
    # Sort by importance for better visualization
    feature_importance_sorted = feature_importance.sort_values('Importance', ascending=True)
    
    # Create horizontal bar chart
    bars = axes[1].barh(feature_importance_sorted['Finger'], 
                        feature_importance_sorted['Importance'], 
                        color=colors)
    axes[1].set_xlabel('Importance', fontsize=12, fontweight='bold')
    axes[1].set_title('Finger Importance for Gesture Recognition', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()  # Highest importance at the top
    
    # Add value labels on the bars
    for i, (bar, importance) in enumerate(zip(bars, feature_importance_sorted['Importance'])):
        width = bar.get_width()
        axes[1].text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{importance:.3f}', ha='left', va='center', fontsize=10)
    
    # Add a grid for better readability
    axes[1].grid(axis='x', alpha=0.3, linestyle='--')
    axes[1].set_axisbelow(True)
    
    # Set x-axis limit with some padding for labels
    axes[1].set_xlim(0, feature_importance_sorted['Importance'].max() * 1.15)
    
    plt.tight_layout()
    plt.savefig('xgboost_model_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Additional visualization: Finger importance pie chart
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(feature_importance)))
    
    wedges, texts, autotexts = ax2.pie(feature_importance['Importance'], 
                                        labels=feature_importance['Finger'],
                                        autopct='%1.1f%%',
                                        colors=colors_pie,
                                        startangle=90,
                                        explode=[0.02] * len(feature_importance))
    
    # Style the pie chart
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    ax2.set_title('Finger Importance Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('finger_importance_pie.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save the model
    import joblib
    model_artifacts = {
        'model': xgb_model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_columns': feature_columns,
        'finger_names': finger_names,
        'class_names': class_names,
        'accuracy': accuracy,
        'f1_score': f1_weighted
    }
    joblib.dump(model_artifacts, 'xgboost_gesture_model.pkl')
    print("\n💾 Model saved to 'xgboost_gesture_model.pkl'")
    
    # Save class mapping for reference
    class_mapping = pd.DataFrame({
        'Original_Sign': class_names,
        'Encoded_Value': range(len(class_names))
    })
    class_mapping.to_csv('class_mapping.csv', index=False)
    print("📁 Class mapping saved to 'class_mapping.csv'")
    
    # Save finger importance for reference
    feature_importance.to_csv('finger_importance.csv', index=False)
    print("📁 Finger importance saved to 'finger_importance.csv'")
    
    print("\n✅ Training completed successfully!")
    
    # Show example predictions
    print("\n" + "="*60)
    print("📝 Example Predictions on Test Set:")
    print("="*60)
    n_examples = min(10, len(X_test))
    for i in range(n_examples):
        pred_class = class_names[y_pred[i]]
        true_class = class_names[y_test[i]]
        prob = np.max(xgb_model.predict_proba(X_test_scaled[i:i+1])[0])
        print(f"Sample {i+1}: True={true_class}, Predicted={pred_class}, Confidence={prob:.3f}")
    
    # Additional analysis: Per-class accuracy
    print("\n" + "="*60)
    print("📊 Per-Class Accuracy:")
    print("="*60)
    per_class_accuracy = []
    for i, class_name in enumerate(class_names):
        mask = y_test == i
        if mask.sum() > 0:
            class_accuracy = (y_pred[mask] == i).sum() / mask.sum()
            per_class_accuracy.append((class_name, class_accuracy, mask.sum()))
            print(f"Class {class_name}: {class_accuracy:.4f} (samples: {mask.sum()})")
    
    # Identify problematic classes
    print("\n" + "="*60)
    print("⚠️ Classes with low accuracy (<70%):")
    print("="*60)
    for class_name, acc, count in per_class_accuracy:
        if acc < 0.7:
            print(f"Class {class_name}: {acc:.4f} (samples: {count})")
    
    # Print summary of most important finger
    most_important = feature_importance.iloc[0]
    print("\n" + "="*60)
    print("🎯 Key Insight:")
    print("="*60)
    print(f"The {most_important['Finger']} finger (Flex{most_important['Sensor'][-1]}) is the most important for gesture recognition with {most_important['Importance']:.3f} importance score.")