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
    
    Args:
        historical_data (pd.DataFrame): Historical weather data DataFrame
        
    Returns:
        tuple: (trained_model, scaler, mae, rmse)
    """
    try:
        # Convert date to datetime if it's not already
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        
        # Add seasonal features to help K-NN find similar days
        historical_data['day_of_year'] = historical_data['date'].dt.dayofyear
        
        # Add cyclical features for better seasonal matching
        historical_data['day_sin'] = np.sin(2 * np.pi * historical_data['day_of_year']/365)
        historical_data['day_cos'] = np.cos(2 * np.pi * historical_data['day_of_year']/365)
        
        # Calculate temperature averages
        historical_data['temp_avg'] = (historical_data['max_temp'] + historical_data['min_temp']) / 2

        # Prepare features - using more relevant features for K-NN
        feature_columns = [
            'humidity',
            'precipitation',
            'wind_speed',
            'day_sin',  # Help find seasonally similar days
            'day_cos',  # Help find seasonally similar days
            'temp_avg'  # Recent temperature as a feature
        ]

        # Prepare the data
        X = historical_data[feature_columns]
        y = historical_data[['max_temp', 'min_temp']]

        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        # Scale features - important for K-NN
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Create base K-NN model
        base_model = KNeighborsRegressor(
            weights='distance',  # Use distance-weighted neighbors
            algorithm='auto',    # Automatically choose best algorithm
            n_jobs=-1           # Use all CPU cores
        )
        
        # Expanded parameter grid for K-NN
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # 1 for manhattan_distance, 2 for euclidean_distance
        }
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Create multi-output regressor
        multi_model = MultiOutputRegressor(grid_search)
        
        print("Training K-NN model...")
        print("This might take a few minutes...")
        multi_model.fit(X_train, y_train)

        # Make predictions
        predictions = multi_model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # Get best parameters
        best_params = multi_model.estimators_[0].best_params_
        
        print("\nModel Training Complete!")
        print(f"Mean Absolute Error: {mae:.2f}°F")
        print(f"Root Mean Squared Error: {rmse:.2f}°F")
        print("\nBest Model Parameters:")
        print(f"Number of neighbors (k): {best_params['n_neighbors']}")
        print(f"Weight function: {best_params['weights']}")
        print(f"Distance metric (p): {best_params['p']}")

        return multi_model, scaler, mae, rmse

    except Exception as e:
        print(f"Error in build_model: {str(e)}")
        print("\nDebugging Information:")
        print(f"Input type: {type(historical_data)}")
        if isinstance(historical_data, pd.DataFrame):
            print("\nColumns in data:")
            print(historical_data.columns.tolist())
            print("\nFirst few rows:")
            print(historical_data.head())
        raise