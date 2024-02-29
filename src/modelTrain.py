import numpy as np
import pandas as pd
import time
from prettytable import PrettyTable

# Import models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from storeLoadUtils import load_params, load_model, store_features
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler


def feature_importance(model, all_features):
    # Get feature importances
    feature_importances = model.feature_importances_

    # Create a DataFrame to display feature importances
    df_feature_importance = pd.DataFrame({'Feature': all_features, 'Importance': feature_importances})
    df_feature_importance = df_feature_importance.sort_values(by='Importance', ascending=False)

    # Print the DataFrame
    print(df_feature_importance.columns.tolist())
    return df_feature_importance


def train_model_pipeline(model_name, X_train, X_test, y_train, y_test, all_features):
    model = train_model(model_name, X_train, y_train)
    feature_importance(model, all_features)
    evaluate_model(model, X_test, y_test)



def train_model(model_name, X_train, y_train):
    # Create a Random Forest regression model with the best parameters from the finetuning
    # model_params = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 150}

    # model = load_model(model_name=model_name)
    # if model :
        # print(f'Trained {model_name} model already exists. Loading existing {model_name} model.')        
        # return model

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
    elif model_name == 'SVM':
        model = SVR()
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


def evaluate_model(model, X_test, y_test):

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