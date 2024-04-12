#####################################################################
#   Author      : Ioannis Kontogiorgakis                            #
#   File        : modelTrain.py                                     #
#   Description : This file is responsible for the selected model   #
#                 training and evaluation on the test set.          #
#####################################################################


# Import necessary libraries
import numpy as np
import pandas as pd
import time
from prettytable import PrettyTable

# Import models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from storeLoadUtils import load_params, load_model, store_features, load_features
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# for plots
from plots import plot_actual_pred, residuals_test


def feature_importance(model, all_features):
    """
    Calculate the feature importances of the given model.

    Parameters:
        model (object): Trained regression model.
        all_features (list): List of all feature names.

    Returns:
        DataFrame: DataFrame containing feature importances sorted in descending order.
    """
    # Get feature importances
    feature_importances = model.feature_importances_

    # Create a DataFrame to display feature importances
    df_feature_importance = pd.DataFrame({'Feature': all_features, 'Importance': feature_importances})
    df_feature_importance = df_feature_importance.sort_values(by='Importance', ascending=False)
    
    # get the features sorted according to their significance
    significant_features = df_feature_importance['Feature'].tolist()
    store_features(significant_features)
    return df_feature_importance


def train_model_pipeline(model_name, X_train, X_test, y_train, y_test):
    """
    Train a regression model specified by model_name and evaluate its performance.

    Parameters:
        model_name (str): Name of the regression model ('RF', 'XGB', 'LR', 'SVM').
        X_train (DataFrame): Training features.
        X_test (DataFrame): Test features.
        y_train (Series): Training target.
        y_test (Series): Test target.

    Returns:
        None
    """
    model = train_model(model_name, X_train, y_train)
    
    # Run only once. Features are places in features.txt file
    # feature_importance(model, all_features)

    evaluate_model(model, model_name, X_test, y_test)



def train_model(model_name, X_train, y_train):
    """
    Train a regression model specified by model_name using the training data.

    Parameters:
        model_name (str): Name of the regression model ('RF', 'XGB', 'LR', 'SVM').
        X_train (DataFrame): Training features.
        y_train (Series): Training target.

    Returns:
        object: Trained regression model.
    """

    # Load specified model's parameters.
    model_params = load_params(model_name)

    if not model_params:
        print(f'- {model_name} parameters file does not exist. Exiting..')
        exit(1)


    print('- Train model..')
    if model_name == 'RF':
        model = RandomForestRegressor(**model_params, random_state=42)
    elif model_name == 'XGB':
        model = XGBRegressor(**model_params, random_state=42)
    elif model_name == 'LR':
        model = LinearRegression()
        # Standardize features using StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    # elif model_name == 'SVM':
        # model = SVR()
    # elif model_name == 'CAT':
        # model = CatBoostRegressor(**model_params, random_state=42)

    # start timer
    start_time = time.time()

    # Train the model
    model.fit(X_train, y_train)

    # end timer
    end_time = time.time()
    print(f"[ Execution time: {end_time - start_time} ]")

    return model



def evaluate_model(model, model_name, X_test, y_test):
    """
    Evaluate the performance of the given regression model on the test data.

    Parameters:
        model (object): Trained regression model.
        model_name (str): The name of the trained model (XGB, RF).
        X_test (DataFrame): Test features.
        y_test (Series): Test target.

    Returns:
        None
    """
    print('-------- Model Evaluation on Test set --------')

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = model.score(X_test, y_test)  # R-squared directly from the model
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Print the results
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared: {r2:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.4f}%")
   
    # Create a table instance
    table = PrettyTable()

    # Add columns
    table.field_names = ["MSE", "RMSE", "R-squared", "MAE", "MAPE"]

    # Add rows
    table.add_row(["{:.4f}".format(mse), "{:.4f}".format(rmse),"{:.4f}".format(r2),"{:.4f}".format(mae),"{:.4f}".format(mape)])
    
    # Print the table
    print(table)

    # Evaluate RF model with different plots
    plot_actual_pred(model_name, y_pred, y_test, rmse, mse, r2)
    residuals_test(y_pred, y_test)