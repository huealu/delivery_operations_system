# Import packages

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from geopy.distance import geodesic
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder



def load_data():
    """Load the data from the CSV file."""
    data = pd.read_csv('data.csv')
    return data


def encode_categorical_data(column_names, data):
    """Encode the categorical data."""
    # Apply LabelEncoding 
    label_encoder = LabelEncoder()

    # Fit and transform data
    if isinstance(column_names, list):
       for col in column_names:
          data[col] = label_encoder.fit_transform(data[col])
    else:
       data[column_names] = label_encoder.fit_transform(data[column_names])
    return data

