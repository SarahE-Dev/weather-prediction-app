import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from .data_collection import get_historical_weather, create_observation_template
from .data_processing import process_data
from .model import build_model

class WeatherApp:
    """
    A class representing a Weather Application that can collect historical data,
    train a predictive model, record observations, make predictions, and visualize data.
    """
    def __init__(self):
        """
        Initialize the WeatherApp with necessary attributes and directories.
        """
        # Initialize model-related attributes
        self.model = None  # Placeholder for the trained model
        self.scaler = None  # Placeholder for the scaler used in preprocessing
        self.historical_data = None  # Placeholder for the historical weather data
        self.predictions_history = []  # List to store prediction histories
        self.mae = None  # Mean Absolute Error metric
        self.rmse = None  # Root Mean Square Error metric
        
        # Define data directories and create them if they don't exist
        self.data_dir = 'data'
        self.dirs = [
            'historical',          # Directory for historical data
            'observations/photos', # Directory for observation photos
            'predictions',         # Directory for prediction results
            'visualizations',      # Directory for saved visualizations
            'models'               # Directory for saved models
        ]
        self._create_directories()

    def _create_directories(self):
        """
        Create necessary directories if they don't exist.
        """
        for dir_name in self.dirs:
            # Create the directory path
            dir_path = os.path.join(self.data_dir, dir_name)
            # Create the directory (and any intermediate directories) if it doesn't exist
            os.makedirs(dir_path, exist_ok=True)

    def collect_historical_data(self, city, days):
        """
        Collect historical weather data from an API for a specified city and number of days.
        
        Args:
            city (str): Name of the city to collect data for.
            days (int): Number of past days to collect data.

        Returns:
            dict: Summary statistics of the collected data, including total days,
                  average high temperature, and average low temperature.
        
        Raises:
            Exception: If there is an error during data collection or processing.
        """
        try:
            # Fetch historical weather data using a custom function
            self.historical_data = get_historical_weather(city, days)
            # Process the raw data (e.g., cleaning, feature engineering)
            self.historical_data = process_data(self.historical_data)
            
            # Calculate summary statistics
            return {
                'total_days': len(self.historical_data),
                'avg_high': self.historical_data['max_temp'].mean(),
                'avg_low': self.historical_data['min_temp'].mean()
            }
        except Exception as e:
            # Raise an exception if data collection fails
            raise Exception(f"Error collecting data: {str(e)}")

    def load_historical_data(self):
        """
        Load previously collected historical data from a CSV file.

        Returns:
            bool: True if data is loaded successfully, False otherwise.
        
        Raises:
            Exception: If there is an error during data loading or processing.
        """
        try:
            # Define the path to the historical data file
            file_path = os.path.join(self.data_dir, 'historical', 'weather_history.csv')
            if os.path.exists(file_path):
                # Read the CSV file into a DataFrame
                self.historical_data = pd.read_csv(file_path)
                # Process the data if necessary
                self.historical_data = process_data(self.historical_data)
                return True
            return False
        except Exception as e:
            # Raise an exception if data loading fails
            raise Exception(f"Error loading historical data: {str(e)}")

    def record_observation(self, observation_data, photo=None):
        """
        Record a new weather observation with optional photo upload capability.

        Args:
            observation_data (dict): A dictionary containing observation details
                                     (e.g., temperature, humidity, wind_speed).
            photo: An optional photo file object to be saved with the observation.

        Returns:
            dict: The recorded observation data.
        
        Raises:
            Exception: If there is an error during recording or input validation.
        """
        try:
            # Get current date and time
            date = datetime.now().strftime('%Y-%m-%d')
            time = datetime.now().strftime('%H:%M')
            
            # Create the observation dictionary with date and time
            observation = {
                'date': date,
                'time': time,
                **observation_data,  # Merge with provided observation data
                'photo_filename': ''  # Initialize photo filename
            }
            
            # Handle photo upload if provided
            if photo is not None:
                # Generate a timestamped filename for the photo
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_extension = os.path.splitext(photo.name)[1].lower()
                new_filename = f"weather_{timestamp}{file_extension}"
                
                # Define the destination path for the photo
                dest_path = os.path.join(self.data_dir, 'observations', 'photos', new_filename)
                
                # Save the photo to the destination path
                with open(dest_path, 'wb') as f:
                    f.write(photo.getbuffer())
                    
                # Update the observation with the photo filename
                observation['photo_filename'] = new_filename
            
            # Validate input ranges
            if observation['temperature'] < -100 or observation['temperature'] > 150:
                raise ValueError("Invalid temperature range")
            if observation['humidity'] < 0 or observation['humidity'] > 100:
                raise ValueError("Invalid humidity range")
            if observation['wind_speed'] < 0:
                raise ValueError("Invalid wind speed")
            
            # Save observation to a CSV file
            observation_file = os.path.join(self.data_dir, 'observations', 'observations.csv')
            # Convert observation to DataFrame for saving
            df = pd.DataFrame([observation])
            # Check if the file already exists to handle headers
            header = not os.path.exists(observation_file)
            # Append the new observation to the CSV file
            df.to_csv(observation_file, mode='a', header=header, index=False)
            
            return observation  # Return the recorded observation
            
        except Exception as e:
            # Raise an exception if there is an error during recording
            raise Exception(f"Error recording observation: {str(e)}")

    def view_observation_photos(self):
        """
        Retrieve photos from the observations/photos directory.

        Returns:
            list: A list of dictionaries containing photo data and filenames.
        
        Raises:
            Exception: If there is an error accessing the photos directory.
        """
        try:
            # Define the photos directory path
            photos_dir = os.path.join(self.data_dir, 'observations', 'photos')
            
            if not os.path.exists(photos_dir):
                return []  # Return empty list if directory doesn't exist
                
            photo_data = []
            # Loop through files in the photos directory
            for filename in os.listdir(photos_dir):
                # Check if the file is an image based on the extension
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Construct the full path to the photo
                    photo_path = os.path.join(photos_dir, filename)
                    # Read the photo in binary mode
                    with open(photo_path, 'rb') as f:
                        photo_bytes = f.read()
                    # Append the photo data and filename to the list
                    photo_data.append({
                        'image': photo_bytes,
                        'filename': filename
                    })
            
            return photo_data  # Return the list of photo data
                
        except Exception as e:
            # Raise an exception if there is an error
            raise Exception(f"Error viewing photos: {str(e)}")

    def train_model(self):
        """
        Train the weather prediction model using the historical data.

        Returns:
            dict: Model metrics including Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).
        
        Raises:
            Exception: If historical data is missing or model training fails.
        """
        # Check if historical data is available
        if self.historical_data is None:
            raise Exception("Please collect or load historical data first!")
                
        try:
            # Ensure historical data is in a DataFrame
            if not isinstance(self.historical_data, pd.DataFrame):
                raise Exception("Historical data is not in the correct format")
                
            # Check for required columns in the historical data
            required_columns = ['humidity', 'precipitation', 'wind_speed', 'max_temp', 'min_temp']
            missing_columns = [col for col in required_columns if col not in self.historical_data.columns]
            if missing_columns:
                raise Exception(f"Missing required columns: {missing_columns}")

            # Build the model using a custom function
            self.model, self.scaler, self.mae, self.rmse = build_model(self.historical_data)
            
            # Save the trained model and scaler for future use
            model_dir = os.path.join(self.data_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)
            
            joblib.dump(self.model, os.path.join(model_dir, 'weather_model.joblib'))
            joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.joblib'))
            
            # Save the model metrics with a timestamp
            metrics = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'mae': self.mae,
                'rmse': self.rmse
            }
            
            # Save metrics to a CSV file
            metrics_file = os.path.join(model_dir, 'model_metrics.csv')
            pd.DataFrame([metrics]).to_csv(
                metrics_file,
                mode='a',  # Append to the file
                header=not os.path.exists(metrics_file),
                index=False
            )
            
            return metrics  # Return the model metrics
                
        except Exception as e:
            # Raise an exception if model training fails
            raise Exception(f"Error during model training: {str(e)}")

    def load_model(self):
        """
        Load a previously trained model and its scaler from disk.

        Returns:
            bool: True if the model is loaded successfully, False otherwise.
        
        Raises:
            Exception: If there is an error during model loading.
        """
        try:
            # Define paths to the model and scaler files
            model_dir = os.path.join(self.data_dir, 'models')
            model_path = os.path.join(model_dir, 'weather_model.joblib')
            scaler_path = os.path.join(model_dir, 'scaler.joblib')
            
            # Check if both the model and scaler files exist
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                # Load the model and scaler
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                
                # Load the latest model metrics if available
                metrics_file = os.path.join(model_dir, 'model_metrics.csv')
                if os.path.exists(metrics_file):
                    metrics = pd.read_csv(metrics_file).iloc[-1]
                    self.mae = metrics['mae']
                    self.rmse = metrics['rmse']
                    
                return True  # Model loaded successfully
            return False  # Model files not found
                
        except Exception as e:
            # Raise an exception if model loading fails
            raise Exception(f"Error loading model: {str(e)}")

    def predict_temperature(self):
        """
        Make and save temperature predictions using the trained model.

        Returns:
            dict: Prediction results including predicted high and low temperatures,
                current conditions, and model metrics.
        
        Raises:
            Exception: If the model is not trained or there is no historical data.
        """
        # Ensure the model is loaded or trained
        if self.model is None and not self.load_model():
            raise Exception("Please train or load a model first!")
                
        try:
            # Check if historical data is available for prediction
            if self.historical_data is None or len(self.historical_data) == 0:
                raise Exception("No historical data available for prediction!")

            # Calculate the date for which to make the prediction
            latest_date = pd.to_datetime(self.historical_data['date']).max()
            prediction_date = latest_date + pd.Timedelta(days=1)
            
            # Calculate the latest average temperature
            latest_temp_avg = (
                self.historical_data['max_temp'].iloc[-1] + 
                self.historical_data['min_temp'].iloc[-1]
            ) / 2
            
            # Calculate cyclical features based on the day of the year
            day_of_year = prediction_date.dayofyear
            day_sin = np.sin(2 * np.pi * day_of_year / 365)
            day_cos = np.cos(2 * np.pi * day_of_year / 365)
            
            # Prepare features for prediction
            prediction_features = pd.DataFrame({
                'humidity': [self.historical_data['humidity'].iloc[-1]],
                'precipitation': [self.historical_data['precipitation'].iloc[-1]],
                'wind_speed': [self.historical_data['wind_speed'].iloc[-1]],
                'day_sin': [day_sin],
                'day_cos': [day_cos],
                'temp_avg': [latest_temp_avg]
            })

            # Scale the features using the loaded scaler
            features_scaled = self.scaler.transform(prediction_features)
            # Make predictions using the loaded model
            predictions = self.model.predict(features_scaled)

            # Create a dictionary to store prediction results
            prediction_dict = {
                'date': prediction_date.strftime('%Y-%m-%d'),
                'predicted_high': round(predictions[0][0], 2),
                'predicted_low': round(predictions[0][1], 2),
                'humidity': prediction_features['humidity'].iloc[0],
                'precipitation': prediction_features['precipitation'].iloc[0],
                'wind_speed': round(prediction_features['wind_speed'].iloc[0], 2),
                'prediction_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Save prediction to a CSV file
            predictions_file = os.path.join(self.data_dir, 'predictions', 'predictions.csv')
            prediction_df = pd.DataFrame([prediction_dict])
            
            # Define the column order for consistency
            columns = [
                'date',
                'prediction_timestamp',
                'predicted_high',
                'predicted_low',
                'humidity',
                'precipitation',
                'wind_speed'
            ]

            # Try to read existing predictions and append new prediction
            try:
                existing_predictions = pd.read_csv(predictions_file)
                # Ensure all columns exist
                for col in columns:
                    if col not in existing_predictions.columns:
                        existing_predictions[col] = None
                # Concatenate the new prediction with existing ones
                updated_predictions = pd.concat(
                    [existing_predictions, prediction_df[columns]], 
                    ignore_index=True
                )
            except (FileNotFoundError, pd.errors.EmptyDataError):
                # If no existing file, start with the new prediction
                updated_predictions = prediction_df[columns]

            # Save the updated predictions back to the CSV file
            updated_predictions.to_csv(predictions_file, index=False)

            # Return prediction results along with current conditions and model metrics
            return {
                'prediction': prediction_dict,
                'current_conditions': {
                    'temperature': latest_temp_avg,
                    'humidity': prediction_features['humidity'].iloc[0],
                    'precipitation': prediction_features['precipitation'].iloc[0],
                    'wind_speed': prediction_features['wind_speed'].iloc[0]
                },
                'model_metrics': {
                    'mae': self.mae,
                    'rmse': self.rmse
                }
            }

        except Exception as e:
            # Raise an exception if prediction fails
            raise Exception(f"Error making prediction: {str(e)}")

    def visualize_data(self, viz_type):
        """
        Create and save weather data visualizations based on the specified type.

        Args:
            viz_type (str): Type of visualization to create.
                            Options include "Temperature Trends", "Humidity vs Temperature",
                            "Wind Speed Distribution", "Correlation Matrix", "Temperature Heatmap".
                
        Returns:
            matplotlib.figure.Figure: The created visualization figure.
        
        Raises:
            Exception: If the historical data is unavailable or visualization fails.
        """
        # Ensure historical data is available
        if self.historical_data is None:
            raise Exception("No data available for visualization!")

        # Set plot style for consistency
        plt.style.use('dark_background')
        colors = plt.cm.plasma(np.linspace(0, 1, 5))
        
        try:
            # Create a figure and axis for the plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Choose the visualization based on the type specified
            if viz_type == "Temperature Trends":
                self._plot_temperature_trends(ax, colors)
            elif viz_type == "Humidity vs Temperature":
                self._plot_humidity_temp(ax)
            elif viz_type == "Wind Speed Distribution":
                self._plot_wind_distribution(ax, colors)
            elif viz_type == "Correlation Matrix":
                self._plot_correlation_matrix(ax)
            elif viz_type == "Temperature Heatmap":
                self._plot_temperature_heatmap(ax)
            else:
                raise ValueError(f"Visualization type '{viz_type}' is not supported.")
            
            # Customize plot aesthetics
            fig.patch.set_facecolor('#1C1C1C')
            ax.set_facecolor('#2F2F2F')
            ax.tick_params(colors='white', which='both')
            for spine in ax.spines.values():
                spine.set_color('white')
                    
            plt.tight_layout()
            
            # Save visualization to the visualizations directory
            save_path = os.path.join(
                self.data_dir, 
                'visualizations', 
                f'viz_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            )
            plt.savefig(save_path, facecolor='#1C1C1C', edgecolor='none', bbox_inches='tight')
            
            return fig  # Return the figure object
                
        except Exception as e:
            # Raise an exception if visualization creation fails
            raise Exception(f"Error creating visualization: {str(e)}")

    # Helper methods for visualization
    def _plot_temperature_trends(self, ax, colors):
        """
        Plot temperature trends over time.

        Args:
            ax: The matplotlib Axes object to plot on.
            colors: Color palette for the plot.
        """
        # Extract dates and temperatures
        dates = pd.to_datetime(self.historical_data['date'])
        max_temp = self.historical_data['max_temp']
        min_temp = self.historical_data['min_temp']
        
        # Fill the area between max and min temperatures
        ax.fill_between(dates, max_temp, min_temp, 
                        alpha=0.3, color=colors[0],
                        label='Temperature Range')
        # Plot max and min temperature lines
        ax.plot(dates, max_temp, color=colors[1], 
                label='Max Temp', linewidth=2)
        ax.plot(dates, min_temp, color=colors[3], 
                label='Min Temp', linewidth=2)
        
        # Set plot titles and labels
        ax.set_title('Temperature Trends Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Temperature (°F)')
        plt.xticks(rotation=45)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

    def _plot_humidity_temp(self, ax):
        """
        Plot humidity versus temperature.

        Args:
            ax: The matplotlib Axes object to plot on.
        """
        # Create a scatter plot of humidity vs. max temperature
        scatter = ax.scatter(
            self.historical_data['humidity'], 
            self.historical_data['max_temp'],
            c=pd.to_datetime(self.historical_data['date']).astype(np.int64),
            cmap='plasma', alpha=0.6
        )
        # Add a colorbar to represent time progression
        plt.colorbar(scatter, label='Time Progression')
        # Set plot titles and labels
        ax.set_title('Humidity vs Temperature')
        ax.set_xlabel('Humidity (%)')
        ax.set_ylabel('Temperature (°F)')

    def _plot_wind_distribution(self, ax, colors):
        """
        Plot the distribution of wind speeds.

        Args:
            ax: The matplotlib Axes object to plot on.
            colors: Color palette for the plot.
        """
        # Create a histogram of wind speeds
        sns.histplot(
            data=self.historical_data, x='wind_speed', 
            bins=20, color=colors[2], alpha=0.7, ax=ax
        )
        # Set plot titles and labels
        ax.set_title('Wind Speed Distribution')
        ax.set_xlabel('Wind Speed (mph)')
        ax.set_ylabel('Count')

    def _plot_correlation_matrix(self, ax):
        """
        Plot a correlation matrix of numerical weather variables.

        Args:
            ax: The matplotlib Axes object to plot on.
        """
        # Select numerical columns for correlation analysis
        numeric_cols = self.historical_data.select_dtypes(include=[np.number]).columns
        # Compute the correlation matrix
        correlation_matrix = self.historical_data[numeric_cols].corr()
        # Use a heatmap to visualize the correlations
        sns.heatmap(
            correlation_matrix, annot=True, 
            cmap='plasma', center=0, ax=ax,
            annot_kws={'color': 'white'}
        )
        # Set plot title
        ax.set_title('Correlation Matrix of Weather Variables')

    def _plot_temperature_heatmap(self, ax):
        """
        Plot a heatmap of average temperatures by month and day.

        Args:
            ax: The matplotlib Axes object to plot on.
        """
        # Add 'month' and 'temp_avg' columns to the data
        self.historical_data['month'] = pd.to_datetime(self.historical_data['date']).dt.month
        self.historical_data['temp_avg'] = (
            self.historical_data['max_temp'] + 
            self.historical_data['min_temp']
        ) / 2
        
        # Create a pivot table for the heatmap
        temp_pivot = self.historical_data.pivot_table(
            values='temp_avg',
            index='month',
            columns=pd.to_datetime(self.historical_data['date']).dt.day,
            aggfunc='mean'
        )
        
        # Plot the heatmap
        sns.heatmap(
            temp_pivot, cmap='magma', 
            annot=False, ax=ax
        )
        # Set plot titles and labels
        ax.set_title('Temperature Heatmap by Month and Day')
        ax.set_xlabel('Day of Month')
        ax.set_ylabel('Month')