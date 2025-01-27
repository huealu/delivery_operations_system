import numpy as np
import lightgbm as lgbm

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor


import time


def grid_search_model(model, X_train, y_train):
    """Perform grid search to find the best model."""
    # Create a list of models
    models = [LinearRegression(), 
              DecisionTreeRegressor(), 
              RandomForestRegressor(), 
              LGBMRegressor()]
    
    # Create a list of parameters for grid search
    params = [{}, 
              {'max_depth': [None, 3, 5, 7]}, 
              {'max_depth': [None, 3, 5, 7], 'n_estimators': [200, 300, 500]}, 
              {'learning_rate': [0.001, 0.01, 0.1], 'max_depth': [3, 4, 5, 7], 'n_estimators': [200, 300, 500, 700]}]
    
    # Perform grid search for each model 
    for id, model in enumerate(models):
        start_time = time.time()
        grid_search = GridSearchCV(model, param_grid= params[id], scoring='r2', cv=4, refit='r2')
        grid_search.fit(X_train, y_train)
        end_time = time.time()

        # Print the results 
        print(f"{model.__class__.__name__}:")
        print("Best parameters:", grid_search.best_params_)
        print("Best R2 score:", grid_search.best_score_)
        print("Time taken: ", end_time - start_time)
        print()
    return grid_search.best_estimator_


def LGBM_model(X_train, y_train):
    """Train the model using LightGBM."""
    # Create the model
    model = LGBMRegressor(learning_rate=0.1, max_depth=7, n_estimators=200, metric='rmse')
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model


def print_model_results(model, X_test, y_test):
    """Print the model results."""
    # Predict the test data
    y_pred = model.predict(X_test)
    
    # Print the results
    print("R2 score:", model.score(X_test, y_test))
    print("MAE:", np.mean(np.abs(y_pred - y_test)))
    print("MSE:", np.mean((y_pred - y_test) ** 2))
    print("RMSE:", np.sqrt(np.mean((y_pred - y_test) ** 2)))