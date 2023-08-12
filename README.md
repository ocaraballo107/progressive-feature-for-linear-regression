# Progressive Feature Selection for Linear Regression

This repository contains Python code that implements a "Progressive Feature Selection" approach for solving Linear Regression problems. The approach involves evaluating each feature separately and then combining the best-performing features to build an optimized Linear Regression model.

## Background

Linear Regression is a common machine learning technique used for predicting a continuous target variable based on one or more input features. Feature selection plays a crucial role in building accurate and interpretable regression models. This approach focuses on systematically selecting and combining features to improve the model's predictive performance. *The files Project-2.ipynb and  Project_II.pdf and Project-2-Opt include the process for the first attempt to solve this problem - just added them for context and reference.* The final version of the code found on this repo is the second version of it with many improvements presented on **progressive-feature-linear-regression.py.**

## Requirements

- Python 3.x
- pandas
- scikit-learn

## Usage

1. Clone this repository:

   ```
   git clone https://github.com/ocaraballo107/progressive-feature-selection.git
   ```

2. Install the required packages:

   ```
   pip install pandas scikit-learn
   ```

3. Prepare your dataset:

   Replace `'your_dataset.csv'` with the path to your dataset CSV file. Ensure that the dataset has columns for each feature (independent variables) and the target variable (dependent variable).

4. Run the code:

   ```
   python progressive_feature_selection.py
   ```

5. Interpret the results:

   The script will iterate through different combinations of features, training and evaluating Linear Regression models for each combination. It will identify the best-performing feature or feature combination based on Mean Squared Error (MSE) on the test set. The selected features and the associated MSE will be printed in the terminal.

## Contributing

Feel free to contribute by opening issues or submitting pull requests. This project welcomes improvements, bug fixes, and additional features.

## License

This project is licensed under the MIT License.
