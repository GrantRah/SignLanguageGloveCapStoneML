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

# 150 flowers consisting of three types

iris = load_iris()      # Loads Iris data and stores in variable (Numerical 2D Array 150x4, Row ~ samples, Columns ~ Slength, SWidth, PLenght, PWidth)
x = iris.data           # Extracts Feature data (Sepal Length, Sepal width, ...) (Rows) for all 3 flowers, [150,4] into x
y = iris.target         # Extracts target labels into y, What the Model is trying/learning to predict 
print('x = ',x[148])    # measurements of flowers[0-149] (Sepal Length, Sepal width, ...) (cm),  for sampele 148
print('y = ',y[148])    # 1D array (150,)  0-49 ~ flower1, 50-99 ~ flower2, 100-149 ~ flower3             
                        # .data always returns a 2D array [number of samples, number of features] (inputs)
                        # .target contains the labels (outputs) corresponding to each row in data

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


"""
# Confusion Matrix with sns heatmap
from sklearn.metrics import ConfusionMatrixDisplay   # Imports class to visualize a 2D Confusion Matrix as heat map
from sklearn.metrics import confusion_matrix         # Imports function to compute a 2D Confusion Matrix

cm = confusion_matrix(y_test, y_pred)  # computes the confusion matrix based on true (tested) outputs and predicted outputs
 
class_names = ['Setosa', 'Versicolor', 'Virginica']  # names of the three iris flower classes

sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names)  
# creates a heatmap using seaborn to visualize the confusion matrix
# True ~ numbers in boxes, d ~ integer format (digits), Reds ~ color scheme, x/yticklabels ~ class names for x and y axes

plt.ylabel('True Label')  
plt.xlabel('Predicted Label')  
plt.title('Confusion Matrix')  
plt.show()  # displays the plot
#print("Confusion Matrix:\n", cm) # prints CMatrix in text form
"""

"""
# Feature Importances, Random Forest allows us to see which input features (Sepal/Petal Length/Width) contributed most to predictions

# rf. is our model, feature_importances_ gives importance scores for each feature in a 1D array, autoatically calculated during training
importances = rf.feature_importances_   # EX: [0.1, 0.2, 0.5, 0.2] ~ importance scores for each of the 4 features, the third contributes 50% to the model's decisions

# iris is our dataset, feature_names gives names to the columns in x data (iris.data)
feature_names = iris.feature_names      # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# Display as a simple text output
for name, importance in zip(feature_names, importances): # zip takes two lists and pairs corresponding elements together
    print(f"{name}: {importance:.4f}") # prints (f string) feature name and its importance score formatted to 4 decimal places

plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=feature_names, palette='viridis')
plt.title("Feature Importances in Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
"""

"""
# Hyperparameter Tuning with GridSearchCV using cross validation
# helps optimize, finding the best combination of parameters for the Random Forest model 
# you pick parameters explcitly or arrange (loop) 
from sklearn.model_selection import GridSearchCV # Imports a class to tune for optimal using grid search with cross-validation

# Define a parameter grid to search over
param_grid = {
    'n_estimators': [50, 100, 200], # number of trees to check
    'max_depth': [None, 2, 4, 6],   # maximum depth of each tree, depth ~ levels for the tree, 2 ~ root + 2 levels (7 total nodes (the circles))
    'min_samples_split': [2, 5, 10] # minimum samples required to split a node to make 2 new ones
}                                   # so if 2 ~ nodes that have 2 or more samples can split and grow until max depth is reached or the node has <2 samples

# Initialize a base Random Forest model
rf = RandomForestClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,           # rf is the model were tuning
    param_grid=param_grid,  # parameters to try(n_estimators, max_depth, min_samples_split)
    cv=5,                   # train data spit into 5 parts(folds) 4 for training and 1 for validation, folds cycle so each will be validation once
                            # 10 sampels → 5 folds of 2 samples each. scores are averaged across all 5 folds
    scoring='accuracy',     # string ~ accuacy = correct predictions / total predictions, sklearn sees 'accuracy' → uses the accuracy_score function internally
    n_jobs=-1               # use all CPU cores ~ I am speed
)

# Fit on the training data
grid_search.fit(x_train, y_train) # runs/performs the grid serac and finds best parameters

# Best hyperparameters
print("Best parameters found:", grid_search.best_params_) # prints the best combination of hyperparameters found during the grid search
print("Best CV Accuracy:", grid_search.best_score_)       # prints the cross-validation accuracy of the best model (one with best parameters)

# Best model
best_rf = grid_search.best_estimator_ # retrieves the best Random Forest model with the best parameters found during the grid search

# Evaluate on test set
y_pred = best_rf.predict(x_test)                      # predicts based on x_test using the best model (were testing our best model)
accuracy = accuracy_score(y_test, y_pred)             # checks accuracy based on test outputs and prediction outputs
print(f"Test Accuracy with GridSearchCV: {accuracy}") # prints accuracy as a float
"""
