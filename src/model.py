from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

def build_model(historical_data):
    """
    Build and train an improved K-NN weather prediction model.

    This function preprocesses historical weather data, adds seasonal features,
    scales the data, performs hyperparameter tuning using GridSearchCV,
    and trains a multi-output K-NN regression model to predict both maximum and minimum temperatures.

    Args:
        historical_data (pd.DataFrame): Historical weather data DataFrame.
            Expected columns: 'date', 'max_temp', 'min_temp', 'humidity', 'precipitation', 'wind_speed'.

    Returns:
        tuple: A tuple containing:
            - trained_model (MultiOutputRegressor): The trained multi-output regression model.
            - scaler (StandardScaler): The scaler fitted on the feature data, for future transformations.
            - mae (float): Mean Absolute Error of the model on the test set.
            - rmse (float): Root Mean Squared Error of the model on the test set.

    Raises:
        ValueError: If there is a data validation error.
        TypeError: If there is a data type mismatch.
        KeyError: If required columns are missing in the data.
        MemoryError: If the system runs out of memory during model training.
        Exception: For any other unexpected errors during model building.
    """
    try:
        # Ensure the 'date' column is in datetime format
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        
        # Add 'day_of_year' to capture seasonal patterns (ranges from 1 to 365/366)
        historical_data['day_of_year'] = historical_data['date'].dt.dayofyear
        
        # Create cyclical features to model the seasonality
        # These help the model to understand the cyclical nature of seasons
        historical_data['day_sin'] = np.sin(2 * np.pi * historical_data['day_of_year'] / 365)
        historical_data['day_cos'] = np.cos(2 * np.pi * historical_data['day_of_year'] / 365)
        
        # Calculate the average temperature as an additional feature
        historical_data['temp_avg'] = (historical_data['max_temp'] + historical_data['min_temp']) / 2

        # Define the feature columns to be used in the model
        feature_columns = [
            'humidity',
            'precipitation',
            'wind_speed',
            'day_sin',     # Seasonal feature
            'day_cos',     # Seasonal feature
            'temp_avg'     # Recent average temperature
        ]
        # I think that the temp_avg is probably the most important feature, that and the seasonal data.
        # Select features and target variables
        X = historical_data[feature_columns]  # Feature matrix
        y = historical_data[['max_temp', 'min_temp']]  # Target variables (multi-output)

        # Handle any missing values by filling them with the mean of each column
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        # Scale the features to standardize the range
        # Important for K-NN because it is sensitive to the scale of the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data into training and testing sets
        # Using 80% of the data for training and 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Initialize the base K-NN regressor
        base_model = KNeighborsRegressor(
            weights='distance',  # Weight neighbors by the inverse of their distance
            algorithm='auto',    # Automatically select the best algorithm based on the data
            n_jobs=-1            # Use all available CPU cores for computation
        )
        
        # Define the parameter grid for hyperparameter tuning
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],  # Different K values
            'weights': ['uniform', 'distance'],       # Uniform or distance-based weighting
            'p': [1, 2]                               # Manhattan (1) or Euclidean (2) distance
        }
        
        # Set up GridSearchCV for cross-validation and hyperparameter tuning
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,                          # 5-fold cross-validation
            scoring='neg_mean_squared_error',  # Use negative MSE as the scoring metric
            n_jobs=-1,                     # Use all CPU cores
            verbose=1                      # Verbosity mode; set to 0 for no output
        )
        
        # Wrap the grid search within a MultiOutputRegressor to handle multiple target variables
        multi_model = MultiOutputRegressor(grid_search)
       
        # Fit the model on the training data
        # This will perform the grid search and train the final model
        multi_model.fit(X_train, y_train)

        # Predict on the test data
        predictions = multi_model.predict(X_test)

        # Calculate regression metrics
        mae = mean_absolute_error(y_test, predictions)  # Mean Absolute Error
        rmse = np.sqrt(mean_squared_error(y_test, predictions))  # Root Mean Squared Error

        # Optionally retrieve the best parameters for each target variable
        # For insight into the chosen hyperparameters after grid search
        best_params = [estimator.best_params_ for estimator in multi_model.estimators_]
        # Uncomment the following line to print best parameters
        # print("Best parameters found:", best_params)

        # Return the trained model, scaler, and performance metrics
        return multi_model, scaler, mae, rmse

    except ValueError as ve:
        # Handle errors related to invalid data values
        raise ValueError(f"Data validation error: {str(ve)}. Please check your input data format.")
        
    except TypeError as te:
        # Handle type-related errors (e.g., non-numeric data)
        raise TypeError(f"Data type error: {str(te)}. Please ensure all features are numeric.")
        
    except pd.errors.EmptyDataError:
        # Handle case where input data is empty
        raise ValueError("The historical data DataFrame is empty.")
        
    except KeyError as ke:
        # Handle missing columns in the data
        raise KeyError(f"Missing required column: {str(ke)}. Please ensure all required features are present.")
        
    except MemoryError:
        # Handle memory errors during computation
        raise MemoryError("Not enough memory to train the model. Try reducing the data size or freeing up memory.")
        
    except Exception as e:
        # Handle any other exceptions that may occur
        raise Exception(f"Unexpected error during model building: {str(e)}")