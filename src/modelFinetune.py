from modelParams import RF_param_grid, CAT_param_grid, XGB_param_grid, LogR_param_grid
from sklearn.model_selection import GridSearchCV
from storeLoadUtils import store_model, store_params
# Import models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.svm import SVR


def model_finetuning(model_name, X_train, X_test, y_train, y_test):

    if model_name == 'RF':
        model = RandomForestRegressor(random_state=42)
        model_param_grid = RF_param_grid
    elif model_name == 'XGB':
        model = XGBRegressor(objective='reg:squarederror', random_state=42)
        model_param_grid = XGB_param_grid
        # elif model_name == 'CAT':
        # model = CatBoostRegressor(random_state=42)
        # model_param_grid = CAT_param_grid




    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=model_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=10)

    # Fit the GridSearchCV object to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best model
    best_params = grid_search.best_params_
    # best_model = grid_search.best_estimator_

    # Print the best parameters
    print(f"{model_name} best Hyperparameters:")
    print(best_params)

    # store best_params
    store_params(f'{model_name}', best_params)

    return best_params