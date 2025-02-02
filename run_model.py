# Import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from geopy.distance import geodesic
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from preprocessing_data import run_preprocessing_data, preprocess_data, categorize_datatype, categorize_datatype_by_dummies
import utils
import base_models
import dl_model


def run_model(train= True, mode='MLP', scaler=False):
    """Run the model."""
    if train: 
        # Load the data
        data = utils.load_data('./data/train.csv')

        # Preprocess the data
        data = preprocess_data(data)

        # Create data for traing and test
        X = data[['Type_of_vehicle', 
                'multiple_deliveries', 
                'Festival', 'City', 
                'weather_condition', 
                'traffic_density', 
                'Vehicle_condition', 
                'day_of_week', 
                'distance']]
        y = data['delivery_time']

        # Categorize the data
        # data = categorize_data(data)

        if scaler:
            # Categorize the datatype by dummies
            X = categorize_datatype_by_dummies(['Type_of_vehicle', 
                                                'multiple_deliveries', 
                                                'Festival', 'City', 
                                                'weather_condition', 
                                                'traffic_density', 
                                                'Vehicle_condition', 
                                                'day_of_week'], X)
        else:
            # Categorize the datatype
            X = utils.encode_categorical_data(['Type_of_vehicle', 
                                    'multiple_deliveries', 
                                    'Festival', 'City', 
                                    'weather_condition', 
                                    'traffic_density', 
                                    'Vehicle_condition', 
                                    'day_of_week'], X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if scaler:
            # Standardize the data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Grid search for the best model
        if mode == 'grid_search': 
            grid_search_model = base_models.grid_search_model(X_train, y_train)
        elif mode == 'LGBM':
            # Run LGBM model
            # model = base_models.LGBM_model(X_train, y_train)

            # Print model results
            # base_models. print_model_results(model, X_train, y_train, X_test, y_test)
            
            # Print feature importance plot
            # base_models.print_feature_importance_plot(model, X_train)
            print("LGBM model is not implemented yet.")
        elif mode == 'MLP':
            # Run MLP model
            print('Start training MLP model')
            model = base_models.MLP_model(X_train, y_train)

            # Print model results
            base_models. print_model_results(model, X_train, y_train, X_test, y_test)
        elif mode == 'DL':
            return dl_model.run_dl_model(train)
        
        # Predict the test data
        y_pred_test = model.predict(X_test)
        
        # Print histplot
        print('Print histplot')
        df = pd.DataFrame({'y_test': y_test, 'y_pred_test': y_pred_test})
        utils.print_histplot(df)

        # Print scatterplot with regression line
        print('Print scatterplot with regression line')
        utils.print_scatterplot_with_regression_line(y_test, y_pred_test)
        
        # Save model
        pickle.dump(model, open(mode+'_model.sav', 'wb')) 
    else:
        # Load the model 
        model = pickle.load(open(mode+'_model.sav', 'rb')) 

    return model 


if __name__ == "__main__":
    """
    For the deep learning model, scaler = True, mode = 'DL'
    For the base models, scaler = False, mode = 'MLP'
    """
    run_model(train=True, mode='MLP', scaler=False)

