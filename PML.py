import pandas as pd                 # Data Handling/Manipulation
import numpy as np                  # Math Operations/Arrays
import matplotlib.pyplot as plt     # plotting basics
import seaborn as sns               # Prettiter Statistical Plots
import sklearn                      # Machine Learing Modles/Tools
import joblib                       # Model Saving/Loading
print("all good")



from sklearn.ensemble import RandomForestClassifier  # Imports the RandomForest ML model
from sklearn.datasets import load_iris               # Imports sklearn's sample dataset (the Iris flower dataset)
from sklearn.model_selection import train_test_split # Imports a function to split the data into a training set (to train the model) and a test set (to test the model)
from sklearn.metrics import accuracy_score           # Imports a function to measure the accuracy
# New stuff below is good?  
from sklearn.preprocessing import StandardScaler     # Imports a function to standardize the data
from sklearn.preprocessing import MinMaxScaler       # Imports a function to normalize the data
from sklearn.model_selection import GridSearchCV     # Imports a function to perform hyperparameter tuning using grid search with cross-validation
from sklearn.model_selection import cross_val_score  # Imports a function to evaluate a model using cross-validation

# 150 flowers consisting of three types

iris = load_iris()      # Loads Iris data and stores in variable (Numerical 2D Array 150x4, Row ~ samples, Columns ~ Slength, SWidth, PLenght, PWidth)
x = iris.data           # Extracts Feature data (Sepal Length, Sepal width, ...) (Rows) for all 3 flowers, [150,4] into x
y = iris.target         # Extracts target labels into y, What the Model is trying/learning to predict 
print('x = ',x[148])    # measurements of flowers[0-149] (Sepal Length, Sepal width, ...) (cm),  for sampele 148
print('y = ',y[148])    # 1D array (150,)  0-49 ~ flower1, 50-99 ~ flower2, 100-149 ~ flower3             
                        # .data always returns a 2D array [number of samples, number of features] (inputs)
                        # .target contains the labels (outputs) corresponding to each row in data
scaler = StandardScaler()  # initialize standard scaler
 

# train_test_split - splits the data into a training and testing set
# Takes in the input (x) and output (y) data. test_size=0.3 → 30% for testing, so 70% is for training
# random_state=42 → it will always pick the same 45 (0.3 * 150) test samples.
# The number is just a specific shuffle seed, so the train/test sets are always the same
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 42) 
x_train_new= scaler.fit_transform(x_train) # standardize the x data (Uses fit transform to scale the training data)
x_test_new= scaler.transform(x_test) # standardize the x data (Uses transform to scale the test data)
# n_estimators - number of trees in the forest, more trees mean longer to train but higher accuracy
# Bootstrapping: Each tree is trained on a random subset of the data
# Feature selection: Each tree splits using a random subset of features at each node
# Setting random_state=42 ensures you get the same forest every time you run the code
rf = RandomForestClassifier(n_estimators = 100, random_state=42) # initialize random forest model

rf.fit(x_train_new, y_train) # .fit trains our ML Model

y_pred = rf.predict(x_test_new) # predicts based on x_test, y_pred = 1D array with 45 predictions [1, 2, 1, 0, ....]
accuracy = accuracy_score(y_test, y_pred) # checks accuracy based on test and prediction outputs
print(f'Accuracy: {accuracy}')
# x.describe()
