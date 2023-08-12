import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the train and test datasets
df_train = pd.read_csv('./datasets/train.csv')
df_test = pd.read_csv('./datasets/test.csv')

# Define the target variable
target = 'SalePrice'

# List of features to consider
all_features = ['Lot Area', 'Overall Qual', 'Year Built', 'Overall Cond', '1st Flr SF', '2nd Flr SF']

# Initialize the selected features list
selected_features = []
best_mse = np.inf

# Iterate through all possible features
for feature in all_features:
    # Add the new feature to the selected features list
    selected_features.append(feature)
    
    # Prepare the data with selected features
    X_train = df_train[selected_features]
    y_train = df_train[target]
    
    # Initialize and train the linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr.predict(X_train)
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_train, y_pred)
    
    # If the current model has lower MSE, update the best features and best MSE
    if mse < best_mse:
        best_features = selected_features.copy()
        best_mse = mse
    
    print(f"Features: {selected_features}, MSE: {mse}")

print("Best features:", best_features)
print("Best MSE:", best_mse)
