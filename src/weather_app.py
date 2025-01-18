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
    def __init__(self):
        """Initialize the WeatherApp with necessary attributes and directories."""
        self.model = None
        self.scaler = None
        self.historical_data = None
        self.predictions_history = []
        self.mae = None
        self.rmse = None
        
        # Create necessary directories
        self.data_dir = 'data'
        self.dirs = [
            'historical',
            'observations/photos',
            'predictions',
            'visualizations',
            'models'
        ]
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for dir_name in self.dirs:
            os.makedirs(os.path.join(self.data_dir, dir_name), exist_ok=True)

    def collect_historical_data(self):
        """Collect historical weather data from API."""
        try:
            city = input("Enter the city name: ").strip()
            days = int(input("Enter the number of days to collect data for: "))
            print("\nCollecting historical data...")
            
            self.historical_data = get_historical_weather(city, days)
            
            # Process the collected data
            self.historical_data = process_data(self.historical_data)
            
            print(f"Historical data for {city} collected successfully!")
            
            # Display basic statistics
            print("\nData Summary:")
            print(f"Total days collected: {len(self.historical_data)}")
            print("\nTemperature Statistics (°F):")
            print(f"Average High: {self.historical_data['max_temp'].mean():.1f}")
            print(f"Average Low: {self.historical_data['min_temp'].mean():.1f}")
            
        except ValueError:
            print("Invalid input. Please enter a valid number of days.")
        except Exception as e:
            print(f"Error: {e}")

    def load_historical_data(self):
        """Load previously collected historical data."""
        try:
            file_path = os.path.join(self.data_dir, 'historical', 'weather_history.csv')
            if os.path.exists(file_path):
                self.historical_data = pd.read_csv(file_path)
                self.historical_data = process_data(self.historical_data)
                print("Historical data loaded successfully!")
                return True
            print("No historical data file found.")
            return False
        except Exception as e:
            print(f"Error loading historical data: {e}")
            return False

    def record_observation(self):
        """Record a new weather observation with photo upload capability."""
        try:
            print("\n=== Weather Observation Recording ===")
            
            # Get the basic observation data
            observation = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': datetime.now().strftime('%H:%M'),
                'temperature': float(input("Current temperature (°F): ")),
                'humidity': float(input("Current humidity (%): ")),
                'cloud_cover': input("Cloud Cover (clear, partly, cloudy): ").strip().lower(),
                'wind_speed': float(input("Wind Speed (mph): ")),
                'wind_direction': input("Wind Direction (N, S, E, W, etc.): ").strip().upper(),
                'precipitation': input("Precipitation (yes/no): ").strip().lower(),
                'pressure': float(input("Atmospheric Pressure (hPa): ")),
                'visibility': float(input("Visibility (miles): ")),
                'notes': input("Additional notes (optional): "),
                'photo_filename': ''
            }

            # Handle photo upload
            upload_photo = input("\nWould you like to upload a photo? (yes/no): ").strip().lower()
            if upload_photo == 'yes':
                while True:
                    photo_path = input("Enter the path to your photo file: ").strip()
                    
                    if os.path.exists(photo_path):
                        # Generate unique filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        file_extension = os.path.splitext(photo_path)[1].lower()
                        new_filename = f"weather_{timestamp}{file_extension}"
                        
                        # Define the destination path
                        dest_path = os.path.join(self.data_dir, 'observations', 'photos', new_filename)
                        
                        try:
                            # Copy the photo to the observations/photos directory
                            import shutil
                            shutil.copy2(photo_path, dest_path)
                            
                            # Update the observation with the photo filename
                            observation['photo_filename'] = new_filename
                            print(f"Photo saved successfully as: {new_filename}")
                            break
                        except Exception as e:
                            print(f"Error saving photo: {e}")
                            retry = input("Would you like to try again? (yes/no): ").strip().lower()
                            if retry != 'yes':
                                break
                    else:
                        print("File not found!")
                        retry = input("Would you like to try again? (yes/no): ").strip().lower()
                        if retry != 'yes':
                            break
            
            # Validate inputs
            if observation['temperature'] < -100 or observation['temperature'] > 150:
                raise ValueError("Invalid temperature range")
            if observation['humidity'] < 0 or observation['humidity'] > 100:
                raise ValueError("Invalid humidity range")
            if observation['wind_speed'] < 0:
                raise ValueError("Invalid wind speed")
            
            # Save observation
            observation_file = os.path.join(self.data_dir, 'observations', 'observations.csv')
            df = pd.DataFrame([observation])
            header = not os.path.exists(observation_file)
            df.to_csv(observation_file, mode='a', header=header, index=False)
            
            print("\nObservation recorded successfully!")
            if observation['photo_filename']:
                print(f"Photo saved as: {observation['photo_filename']}")
            
        except ValueError as ve:
            print(f"Invalid input: {ve}")
        except Exception as e:
            print(f"Error: {e}")

    def view_observation_photos(self):
        """View photos associated with weather observations."""
        try:
            photos_dir = os.path.join(self.data_dir, 'observations', 'photos')
            observation_file = os.path.join(self.data_dir, 'observations', 'observations.csv')
            
            if not os.path.exists(observation_file):
                print("No observations recorded yet!")
                return
                
            observations = pd.read_csv(observation_file)
            observations_with_photos = observations[observations['photo_filename'].notna() & 
                                                (observations['photo_filename'] != '')]
            
            if len(observations_with_photos) == 0:
                print("No photos found in observations!")
                return
                
            print("\n=== Weather Observation Photos ===")
            for idx, obs in observations_with_photos.iterrows():
                print(f"\nObservation from {obs['date']} {obs['time']}")
                print(f"Temperature: {obs['temperature']}°F")
                print(f"Conditions: {obs['cloud_cover']}")
                print(f"Photo: {obs['photo_filename']}")
                
                # Option to open the photo
                photo_path = os.path.join(photos_dir, obs['photo_filename'])
                if os.path.exists(photo_path):
                    open_photo = input("Would you like to open this photo? (yes/no): ").strip().lower()
                    if open_photo == 'yes':
                        try:
                            import webbrowser
                            webbrowser.open(photo_path)
                        except Exception as e:
                            print(f"Error opening photo: {e}")
                else:
                    print("Photo file not found!")
                    
        except Exception as e:
            print(f"Error viewing photos: {e}")

    def train_model(self):
        """Train the weather prediction model."""
        if self.historical_data is None:
            print("Please collect or load historical data first!")
            return
                
        try:
            # Verify data is properly formatted
            if not isinstance(self.historical_data, pd.DataFrame):
                print("Error: Historical data is not in the correct format")
                return
                
            # Verify required columns exist
            required_columns = ['humidity', 'precipitation', 'wind_speed', 'max_temp', 'min_temp']
            missing_columns = [col for col in required_columns if col not in self.historical_data.columns]
            if missing_columns:
                print(f"Error: Missing required columns: {missing_columns}")
                return

            # Train the model
            self.model, self.scaler, self.mae, self.rmse = build_model(self.historical_data)
            
            # Save the model and scaler
            model_dir = os.path.join(self.data_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)
            
            joblib.dump(self.model, os.path.join(model_dir, 'weather_model.joblib'))
            joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.joblib'))
            
            # Save model metrics
            metrics = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'mae': self.mae,
                'rmse': self.rmse
            }
            
            metrics_file = os.path.join(model_dir, 'model_metrics.csv')
            pd.DataFrame([metrics]).to_csv(
                metrics_file,
                mode='a',
                header=not os.path.exists(metrics_file),
                index=False
            )
            
            print(f"\nModel trained successfully!")
            print(f"Mean Absolute Error: {self.mae:.2f}°F")
            print(f"Root Mean Squared Error: {self.rmse:.2f}°F")
            
        except Exception as e:
            print(f"Error during model training: {e}")

    def load_model(self):
        """Load a previously trained model."""
        try:
            model_dir = os.path.join(self.data_dir, 'models')
            model_path = os.path.join(model_dir, 'weather_model.joblib')
            scaler_path = os.path.join(model_dir, 'scaler.joblib')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                
                # Load latest metrics
                metrics_file = os.path.join(model_dir, 'model_metrics.csv')
                if os.path.exists(metrics_file):
                    metrics = pd.read_csv(metrics_file).iloc[-1]
                    self.mae = metrics['mae']
                    self.rmse = metrics['rmse']
                    
                print("Model loaded successfully!")
                return True
            else:
                print("No saved model found. Please train a new model.")
                return False
                
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict_temperature(self):
        """Make temperature predictions using the K-NN model and save to CSV."""
        if self.model is None:
            if not self.load_model():
                print("Please train a model first!")
                return
                
        try:
            if self.historical_data is None or len(self.historical_data) == 0:
                print("No historical data available for prediction!")
                return

            print("\nPredicting tomorrow's temperatures...")

            # Get the most recent data
            latest_date = pd.to_datetime(self.historical_data['date']).max()
            prediction_date = latest_date + pd.Timedelta(days=1)
            
            # Calculate temp_avg from the most recent max and min temperatures
            latest_temp_avg = (self.historical_data['max_temp'].iloc[-1] + 
                            self.historical_data['min_temp'].iloc[-1]) / 2
            
            # Calculate day of year and seasonal features
            day_of_year = prediction_date.dayofyear
            day_sin = np.sin(2 * np.pi * day_of_year/365)
            day_cos = np.cos(2 * np.pi * day_of_year/365)
            
            # Create features for prediction
            prediction_features = pd.DataFrame({
                'humidity': [self.historical_data['humidity'].iloc[-1]],
                'precipitation': [self.historical_data['precipitation'].iloc[-1]],
                'wind_speed': [self.historical_data['wind_speed'].iloc[-1]],
                'day_sin': [day_sin],
                'day_cos': [day_cos],
                'temp_avg': [latest_temp_avg]
            })

            # Scale features
            features_scaled = self.scaler.transform(prediction_features)

            # Make prediction
            predictions = self.model.predict(features_scaled)

            # Create prediction dictionary
            prediction_dict = {
                'date': prediction_date.strftime('%Y-%m-%d'),
                'predicted_high': round(predictions[0][0], 2),
                'predicted_low': round(predictions[0][1], 2),
                'humidity': prediction_features['humidity'].iloc[0],
                'precipitation': prediction_features['precipitation'].iloc[0],
                'wind_speed': round(prediction_features['wind_speed'].iloc[0], 2),
                'prediction_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Save prediction to CSV
            predictions_file = 'data/predictions/predictions.csv'
            os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
            
            # Convert prediction to DataFrame
            prediction_df = pd.DataFrame([prediction_dict])
            
            # Define column order
            columns = [
                'date',
                'prediction_timestamp',
                'predicted_high',
                'predicted_low',
                'humidity',
                'precipitation',
                'wind_speed'
            ]

            try:
                # Try to read existing predictions
                existing_predictions = pd.read_csv(predictions_file)
                
                # Check if all columns exist, add missing ones
                for col in columns:
                    if col not in existing_predictions.columns:
                        existing_predictions[col] = None
                
                # Append new prediction
                updated_predictions = pd.concat([existing_predictions, prediction_df[columns]], ignore_index=True)
                
            except (FileNotFoundError, pd.errors.EmptyDataError):
                # If file doesn't exist or is empty, create new DataFrame
                updated_predictions = prediction_df[columns]

            # Save to CSV
            updated_predictions.to_csv(predictions_file, index=False)

            # Display results
            print(f"\nPrediction for tomorrow ({prediction_date.strftime('%Y-%m-%d')}):")
            print(f"Predicted High: {predictions[0][0]:.1f}°F")
            print(f"Predicted Low: {predictions[0][1]:.1f}°F")
            
            print("\nCurrent Conditions Used:")
            print(f"Temperature: {latest_temp_avg:.1f}°F (average)")
            print(f"Humidity: {prediction_features['humidity'].iloc[0]}%")
            print(f"Precipitation: {'Yes' if prediction_features['precipitation'].iloc[0] == 1 else 'No'}")
            print(f"Wind Speed: {prediction_features['wind_speed'].iloc[0]:.1f} mph")

            if hasattr(self, 'mae') and hasattr(self, 'rmse'):
                print("\nModel Performance Metrics:")
                print(f"Mean Absolute Error: ±{self.mae:.1f}°F")
                print(f"Root Mean Squared Error: ±{self.rmse:.1f}°F")

            print(f"\nPrediction saved to {predictions_file}")

        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            print("\nDebugging Information:")
            print("Available features:", prediction_features.columns.tolist())
            print("Feature values:")
            print(prediction_features)
       

    def visualize_data(self):
        """Visualize weather data with various plots using dark theme and plasma/magma colors."""
        if self.historical_data is None:
            print("No data available for visualization!")
            return

        print("\n=== Data Visualization Options ===")
        print("1. Temperature Trends")
        print("2. Humidity vs Temperature")
        print("3. Wind Speed Distribution")
        print("4. Correlation Matrix")
        print("5. Temperature Heatmap")
        
        choice = input("\nSelect visualization type (1-5): ")
        
        # Set dark theme style
        plt.style.use('dark_background')
        
        # Custom color palette using plasma colors
        colors = plt.cm.plasma(np.linspace(0, 1, 5))
        
        try:
            if choice == '1':
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Create gradient fill for temperature range
                dates = self.historical_data['date']
                max_temp = self.historical_data['max_temp']
                min_temp = self.historical_data['min_temp']
                
                ax.fill_between(dates, max_temp, min_temp, 
                            alpha=0.3, color=colors[0],
                            label='Temperature Range')
                ax.plot(dates, max_temp, color=colors[1], 
                    label='Max Temp', linewidth=2)
                ax.plot(dates, min_temp, color=colors[3], 
                    label='Min Temp', linewidth=2)
                
                ax.set_title('Temperature Trends Over Time', 
                            color='white', pad=20, fontsize=14)
                ax.set_xlabel('Date', color='white')
                ax.set_ylabel('Temperature (°F)', color='white')
                plt.xticks(rotation=45)
                ax.legend(facecolor='black', edgecolor='white')
                
                # Grid styling
                ax.grid(True, linestyle='--', alpha=0.3)
                
            elif choice == '2':
                fig, ax = plt.subplots(figsize=(12, 6))
                
                scatter_max = ax.scatter(self.historical_data['humidity'], 
                                    self.historical_data['max_temp'],
                                    c=self.historical_data['date'].astype(np.int64),
                                    cmap='plasma', alpha=0.6, label='Max Temp')
                scatter_min = ax.scatter(self.historical_data['humidity'], 
                                    self.historical_data['min_temp'],
                                    c=self.historical_data['date'].astype(np.int64),
                                    cmap='magma', alpha=0.6, label='Min Temp')
                
                plt.colorbar(scatter_max, label='Time Progression')
                ax.set_title('Humidity vs Temperature', 
                            color='white', pad=20, fontsize=14)
                ax.set_xlabel('Humidity (%)', color='white')
                ax.set_ylabel('Temperature (°F)', color='white')
                ax.legend(facecolor='black', edgecolor='white')
                
            elif choice == '3':
                fig, ax = plt.subplots(figsize=(12, 6))
                
                sns.histplot(data=self.historical_data, x='wind_speed', 
                            bins=20, color=colors[2], alpha=0.7)
                ax.set_title('Wind Speed Distribution', 
                            color='white', pad=20, fontsize=14)
                ax.set_xlabel('Wind Speed (mph)', color='white')
                ax.set_ylabel('Count', color='white')
                
            elif choice == '4':
                fig, ax = plt.subplots(figsize=(10, 8))
                
                numeric_cols = self.historical_data.select_dtypes(include=[np.number]).columns
                correlation_matrix = self.historical_data[numeric_cols].corr()
                
                sns.heatmap(correlation_matrix, annot=True, 
                        cmap='plasma', center=0, ax=ax,
                        annot_kws={'color': 'white'})
                
                ax.set_title('Correlation Matrix of Weather Variables', 
                            color='white', pad=20, fontsize=14)
                
            elif choice == '5':
                # New visualization: Temperature Heatmap by Month and Hour
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Extract month and create average temperature
                self.historical_data['month'] = self.historical_data['date'].dt.month
                self.historical_data['temp_avg'] = (self.historical_data['max_temp'] + 
                                                self.historical_data['min_temp']) / 2
                
                # Create pivot table for heatmap
                temp_pivot = self.historical_data.pivot_table(
                    values='temp_avg',
                    index='month',
                    columns=self.historical_data['date'].dt.day,
                    aggfunc='mean'
                )
                
                sns.heatmap(temp_pivot, cmap='magma', 
                        annot=False, ax=ax)
                ax.set_title('Temperature Heatmap by Month and Day', 
                            color='white', pad=20, fontsize=14)
                ax.set_xlabel('Day of Month', color='white')
                ax.set_ylabel('Month', color='white')
                
            else:
                print("Invalid choice!")
                return

            # Common styling for all plots
            fig.patch.set_facecolor('#1C1C1C')  # Dark background
            ax.set_facecolor('#2F2F2F')  # Slightly lighter background for plot area
            
            # Style the axis labels and ticks
            ax.tick_params(colors='white', which='both')
            for spine in ax.spines.values():
                spine.set_color('white')

            # Save and show the plot
            plt.tight_layout()
            save_path = os.path.join(
                self.data_dir, 
                'visualizations', 
                f'viz_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            )
            plt.savefig(save_path, facecolor='#1C1C1C', edgecolor='none', bbox_inches='tight')
            plt.show()
            print(f"\nVisualization saved to: {save_path}")

        except Exception as e:
            print(f"Error creating visualization: {e}")
            raise

    def export_data(self):
        """Export weather data in various formats."""
        if self.historical_data is None:
            print("No data available to export!")
            return

        print("\n=== Export Options ===")
        print("1. CSV")
        print("2. Excel")
        print("3. JSON")
        
        choice = input("\nSelect export format (1-3): ")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            if choice == '1':
                filename = f'weather_data_{timestamp}.csv'
                self.historical_data.to_csv(filename, index=False)
            elif choice == '2':
                filename = f'weather_data_{timestamp}.xlsx'
                self.historical_data.to_excel(filename, index=False)
            elif choice == '3':
                filename = f'weather_data_{timestamp}.json'
                self.historical_data.to_json(filename, orient='records')
            else:
                print("Invalid choice!")
                return
            
            print(f"Data exported successfully to: {filename}")
        except Exception as e:
            print(f"Error exporting data: {e}")