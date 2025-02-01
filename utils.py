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



def load_data(file_path):
    """Load the data from the CSV file."""
    data = pd.read_csv(file_path)
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


def print_scatterplot_with_regression_line(x, y):
    """Print a scatterplot with regression line."""
    # Create a scatterplot
    plt.scatter(x, y)
    
    # Fit a regression line
    m, b = np.polyfit(x, y, 1)
    
    # Plot the regression line
    plt.plot(x, m * x + b, color='red')
    
    # Add labels and title
    plt.xlabel('y_test')
    plt.ylabel('y_pred_test')
    plt.title('Scatterplot of y_test vs y_pred_test')
    
    # Show the plot
    plt.show()


def print_histplot(data):
    """Print a histogram plot."""
    # Create a histogram
    sns.histplot(data)
    
    # Add labels and title
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {data.columns[0]} and {data.columns[1]}')
    
    # Show the plot
    plt.show()

