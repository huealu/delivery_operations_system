import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd

import utils, preprocessing_data

def create_model(input_shape):
    """Create a model."""
    model = Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1))
    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    """Train the model."""
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size,
                        validation_data=(X_test, y_test))
    return model, history


def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    loss, mae = model.evaluate(X_test, y_test)
    return loss, mae


def run_dl_model():
    """Run the deep learning model."""
    # Load the data
    data = utils.load_data('./data/train.csv')

    # Preprocess the data
    data = preprocessing_data.preprocess_data(data)

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

    # Categorize the datatype
    X = utils.encode_categorical_data(['Type_of_vehicle', 
                                      'multiple_deliveries', 
                                      'Festival', 'City', 
                                      'weather_condition', 
                                      'traffic_density', 
                                      'Vehicle_condition', 
                                      'day_of_week'], X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the model
    model = create_model(X_train.shape[1])

    model.summary()

    # Train the model
    model, history = train_model(model, X_train, y_train, X_test, y_test)

    # Evaluate the model
    loss, mae = evaluate_model(model, X_test, y_test)
    print(f"Loss: {loss}, MAE: {mae}")

    # Predict the test data
    y_pred_test = model.predict(X_test)

    # Print loss history
    utils.print_loss_history(history)

    # Print histplot
    print('Print histplot')
    df = pd.DataFrame({'y_test': y_test, 'y_pred_test': y_pred_test.reshape(-1)})
    utils.print_histplot(df)
    

    # Print scatterplot with regression line
    print('Print scatterplot with regression line')
    utils.print_scatterplot_with_regression_line(y_test, y_pred_test)

    return model


if __name__ == "__main__":
    run_dl_model()