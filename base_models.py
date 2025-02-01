import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV

import time

# import lightgbm as lgbm
# from lightgbm import LGBMRegressor


def grid_search_model(X_train, y_train, model=None):
    """Perform grid search to find the best model."""
    # Create a list of models
    models = [LinearRegression(), 
              DecisionTreeRegressor(), 
              RandomForestRegressor()
              # LGBMRegressor()
              ]
    
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
    """
    # Create the model
    model = LGBMRegressor(learning_rate=0.1, 
                          max_depth=7, 
                          n_estimators=200, 
                          metric='rmse')
    
    # Train the model
    model.fit(X_train, y_train)
    return model
    """
    print("LGBM model is not implemented yet.")


def print_feature_importance_plot(model, X_train):
    """Print a feature importance plot."""

    """
    # Get the feature importances
    # Plot feature importance using Gain
    lgbm.plot_importance(model, importance_type="gain", figsize=(7,6), title="LightGBM Feature Importance (Gain)")

    # Plot feature importance using Split
    lgbm.plot_importance(model, importance_type="split", figsize=(7,6), title="LightGBM Feature Importance (Split)")

    # Plot feature importance 
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.suptitle('Horizontally stacked subplots')
    # ax1 = lgbm.plot_importance(model, importance_type="gain", figsize=(7,6), title="LightGBM Feature Importance (Gain)")
    # ax2 = lgbm.plot_importance(model, importance_type="split", figsize=(7,6), title="LightGBM Feature Importance (Split)")
    
    # Show the plot
    plt.show()
    """
    print("Feature importance plot is not implemented yet due to the lightGBM package issue when installing the package.")


def MLP_model(X_train, y_train):
    """Train the model using Multi-Layer Perceptron."""
    # Create the model
    model = MLPRegressor(hidden_layer_sizes=(100, 50, 25), 
                         max_iter=1000, 
                         alpha=0.01, 
                         learning_rate='adaptive', 
                         random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    return model



def print_model_results(model, X_train, y_train, X_test, y_test):
    """Print the model results."""
    # Generate prediction for train and test data
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate the RMSE for train data
    rmse = root_mean_squared_error(y_train, y_pred_train)
    
    # Print the results
    print("MSE:", np.mean((y_pred_test - y_test) ** 2))
    print("RMSE:", np.sqrt(np.mean((y_pred_test - y_test) ** 2)))
    print("Accuracy:", model.score(X_test, y_test))