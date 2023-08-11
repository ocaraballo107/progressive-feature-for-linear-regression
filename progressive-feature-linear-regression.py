import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from itertools import combinations

# Load your dataset and preprocess it
# ...

# Split the dataset into features (X) and target (y)
X = dataset.drop('cost', axis=1)
y = dataset['cost']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize variables to keep track of the best model
best_mse = float('inf')
best_model = None
best_features = None

# Iterate through different combinations of features
for num_features in range(1, len(X_train.columns) + 1):
    for feature_combination in combinations(X_train.columns, num_features):
        # Create a new DataFrame with the selected feature combination
        feature_combination_X_train = X_train[list(feature_combination)]
        feature_combination_X_test = X_test[list(feature_combination)]
        
        # Initialize and train a linear regression model
        model = LinearRegression()
        model.fit(feature_combination_X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(feature_combination_X_test)
        
        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        
        # Update the best model if this model performs better
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_features = feature_combination

# Print the best feature combination and its associated MSE
print("Best Feature Combination:", best_features)
print("Best MSE:", best_mse)

# You can now use the best model for prediction based on the selected features
