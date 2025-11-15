import pandas as pd                 # Data Handling/Manipulation
import numpy as np                  # Math Operations/Arrays
import matplotlib.pyplot as plt     # plotting basics
import seaborn as sns               # Prettiter Statistical Plots
import sklearn                      # Machine Learing Modles/Tools
print("all good")



from sklearn.ensemble import RandomForestClassifier  # Imports the RandomForest ML model
from sklearn.datasets import load_iris               # Imports sklearn's sample dataset (the Iris flower dataset)
from sklearn.model_selection import train_test_split # Imports a function to split the data into a training set (to train the model) and a test set (to test the model)
from sklearn.metrics import accuracy_score           # Imports a function to measure the accuracy


df = pd.read_csv('winequality-red.csv', delimiter=';')  # Loads the Wine Quality dataset from a CSV file into a pandas DataFrame

x = df.drop('quality', axis=1).values      # Extracts feature data by dropping the 'quality' column, .values converts DataFrame to NumPy array
y = df['quality'].values              # Extracts target labels (quality scores) into y as a NumPy array
print('Dataset shape:', x.shape)
print('First sample features:', x[0])    # First wine sample's measurements
print('First sample quality:', y[0])     # Quality rating of first sample
print('Unique quality values:', np.unique(y))  # See what quality ratings exist
# train_test_split - splits the data into a training and testing set
# Takes in the input (x) and output (y) data. test_size=0.3 → 30% for testing, so 70% is for training
# random_state=42 → it will always pick the same 45 (0.3 * 150) test samples.
# The number is just a specific shuffle seed, so the train/test sets are always the same
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 42) 

# n_estimators - number of trees in the forest, more trees mean longer to train but higher accuracy
# Bootstrapping: Each tree is trained on a random subset of the data
# Feature selection: Each tree splits using a random subset of features at each node
# Setting random_state=42 ensures you get the same forest every time you run the code
rf = RandomForestClassifier(n_estimators = 100, random_state=42) # initialize random forest model

rf.fit(x_train, y_train) # .fit trains our ML Model

y_pred = rf.predict(x_test) # predicts based on x_test, y_pred = 1D array with 45 predictions [1, 2, 1, 0, ....]
accuracy = accuracy_score(y_test, y_pred) # checks accuracy based on test and prediction outputs
print(f'First Accuracy: {accuracy}')


# Confusion Matrix with sns heatmap
from sklearn.metrics import ConfusionMatrixDisplay   
from sklearn.metrics import confusion_matrix         

cm = confusion_matrix(y_test, y_pred)  

# Get the unique quality values and sort them
quality_values = sorted(np.unique(y))
class_names = [f'Quality {q}' for q in quality_values]  # Names like 'Quality 3', 'Quality 4', etc.

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=class_names, yticklabels=class_names)  

plt.ylabel('True Label')  
plt.xlabel('Predicted Label')  
plt.title('Wine Quality Confusion Matrix')  
plt.tight_layout()
plt.show()  

print("Confusion Matrix:\n", cm)
# Feature Importances for wine dataset
importances = rf.feature_importances_   

# Use the column names from your dataframe (excluding 'quality')
feature_names = df.columns[:-1]  # All columns except the last one

# Display as a simple text output
print("Feature Importances:")
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f}")

# Create bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names, palette='viridis')
plt.title("Wine Quality - Feature Importances in Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Hyperparameter Tuning with GridSearchCV
from sklearn.model_selection import GridSearchCV 

# Define a parameter grid to search over
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 15],  # Adjusted for wine dataset complexity
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]    # Added min_samples_leaf for better control
}

# Initialize a base Random Forest model
rf = RandomForestClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fit on the training data
grid_search.fit(x_train, y_train)

# Best hyperparameters
print("Best parameters found:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

# Best model
best_rf = grid_search.best_estimator_

# Evaluate on test set
y_pred = best_rf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy with GridSearchCV: {accuracy}")

# Save and load model with joblib
import joblib    

# Save the trained model
joblib.dump(rf, 'wine_quality_model.pkl')  # Changed filename to be more specific

# Load the model
loaded_model = joblib.load('wine_quality_model.pkl')

# Test the loaded model
y_loaded_pred = loaded_model.predict(x_test)
loaded_accuracy = accuracy_score(y_test, y_loaded_pred)
print(f'Loaded Model Accuracy: {loaded_accuracy}')

# Optional: Save the best model from GridSearchCV too
joblib.dump(best_rf, 'wine_quality_best_model.pkl')
print("Best model saved as 'wine_quality_best_model.pkl'")  


