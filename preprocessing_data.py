# Import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from geopy.distance import geodesic
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler



def clean_extra_space(data):
    """Clean the data by removing extra space"""
    # Remove `(min)` in `Time_taken(min)` column
    data['delivery_time'] = data['Time_taken(min)'].str.replace('(min) ', '').astype('float')
    
    # Remove `conditions` in `Weatherconditions`
    data['weather_condition'] = data['Weatherconditions'].str.replace('conditions', '')
    
    # Remove extra space
    data['traffic_density'] = data['Road_traffic_density'].str.replace(' ', '')
    data['Festival'] = data['Festival'].str.replace(' ', '')
    data['City'] = data['City'].str.replace(' ', '')
    data['Type_of_vehicle'] = data['Type_of_vehicle'].str.replace(' ', '')
    return data


def handle_missing_data(data):
    """Handle the missing data"""
    # Replace `NaN` data by using `np.nan`
    data.replace('NaN', float(np.nan), regex=True, inplace=True)
    # Remove rows with missing values 
    data = data.dropna()
    return data


def convert_date_time_data_type(data):
    """Convert the date time data type"""
    # Convert `Order_Date` to date time
    data['Order_Date'] = pd.to_datetime(data['Order_Date'], format='%d-%m-%Y')
    
    # Calculate different time from order to pickup time
    data['order_to_pickup_time'] = pd.to_timedelta(data['Time_Order_picked']) - pd.to_timedelta(data['Time_Orderd'])
    
    return data


def convert_date_to_day(data):
    """Convert date to day"""
    # Convert date to day
    data['day_of_week'] = data['Order_Date'].dt.day_of_week.astype(int)
    return data


def calculate_distance(data):
    """Calculate the lat and long distance from departure to the arrival point"""
    depart_location = data[['Restaurant_latitude', 'Restaurant_longitude']].to_numpy()
    arrival_location = data[['Delivery_location_latitude', 'Delivery_location_longitude']].to_numpy()
    
    # Caculate the distance between two points
    data['distance'] = np.array([geodesic(start, end).miles for start, end in zip(depart_location, arrival_location)])
    return data


def categorize_data(data):
    """Categorize the data"""
    # Categorize the data
    data['Type_of_vehicle'] = pd.Categorical(data['Type_of_vehicle'])
    data['multiple_deliveries'] = pd.Categorical(data['multiple_deliveries'])
    data['Festival'] = pd.Categorical(data['Festival'])
    data['City'] = pd.Categorical(data['City'])
    data['weather_condition'] = pd.Categorical(data['weather_condition'])
    data['traffic_density'] = pd.Categorical(data['traffic_density'])
    data['Vehicle_condition'] = pd.Categorical(data['Vehicle_condition'])
    data['day_of_week'] = pd.Categorical(data['day_of_week'])
    return data


def categorize_datatype_by_dummies(column_names, data):
    """Categorize datatype to category"""
    if isinstance(column_names, list):
        for col in column_names:
            # Convert datatype to category
            data[col] = data[col].astype('str')
    else:
        data[column_names] = data[column_names].astype('str')
    df = pd.get_dummies(data, drop_first=True)
    return df



def categorize_datatype(column_names, data):
    """Categorize datatype to category"""
    if isinstance(column_names, list):
        for col in column_names:
            # Convert datatype to category
            data[col] = data[col].astype('category')
    else:
        data[column_names] = data[column_names].astype('category')

    return data


def preprocess_data(data):
    """Preprocess the data"""
    # Clean the data
    data = clean_extra_space(data)
    
    # Handle the missing data
    data = handle_missing_data(data)

    # Convert the date time data type
    data = convert_date_time_data_type(data)

    # Convert date to day
    data = convert_date_to_day(data)

    # Calculate the distance
    data = calculate_distance(data)
    
    return data


def run_preprocessing_data(data, scaler=False):
    """Run preprocessing data"""
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
    Y = data['delivery_time']

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
        X = categorize_datatype(['Type_of_vehicle', 
                                 'multiple_deliveries', 
                                 'Festival', 'City', 
                                 'weather_condition', 
                                 'traffic_density', 
                                 'Vehicle_condition', 
                                 'day_of_week'], X)

    
    return data, X, Y