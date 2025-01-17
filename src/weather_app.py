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
        """Make temperature predictions using the improved K-NN model."""
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
            
            # Create features for prediction
            prediction_features = pd.DataFrame({
                'humidity': [self.historical_data['humidity'].iloc[-1]],
                'precipitation': [self.historical_data['precipitation'].iloc[-1]],
                'wind_speed': [self.historical_data['wind_speed'].iloc[-1]],
                'day_sin': [np.sin(2 * np.pi * prediction_date.dayofyear/365)],
                'day_cos': [np.cos(2 * np.pi * prediction_date.dayofyear/365)],
                'temp_avg': [self.historical_data['temp_avg'].iloc[-1]]
            })

            # Scale features
            features_scaled = self.scaler.transform(prediction_features)

            # Make prediction
            predictions = self.model.predict(features_scaled)

            # Find similar days for context
            distances, indices = self.model.estimators_[0].best_estimator_.kneighbors(features_scaled)
            similar_days = self.historical_data.iloc[indices[0]]

            # Display results
            print(f"\nPrediction for tomorrow ({prediction_date.strftime('%Y-%m-%d')}):")
            print(f"Predicted High: {predictions[0][0]:.1f}°F")
            print(f"Predicted Low: {predictions[0][1]:.1f}°F")
            
            print("\nBased on similar historical days:")
            for idx, day in similar_days.iterrows():
                print(f"Date: {day['date'].strftime('%Y-%m-%d')}")
                print(f"Actual High: {day['max_temp']:.1f}°F, Low: {day['min_temp']:.1f}°F")
                print(f"Conditions: Humidity {day['humidity']}%, "
                    f"{'Rain' if day['precipitation'] == 1 else 'No Rain'}, "
                    f"Wind {day['wind_speed']:.1f} mph")
                print()

            if hasattr(self, 'mae') and hasattr(self, 'rmse'):
                print("\nModel Performance Metrics:")
                print(f"Mean Absolute Error: ±{self.mae:.1f}°F")
                print(f"Root Mean Squared Error: ±{self.rmse:.1f}°F")

        except Exception as e:
            print(f"Error making prediction: {str(e)}")
       

    def visualize_data(self):
        """Visualize weather data with various plots."""
        if self.historical_data is None:
            print("No data available for visualization!")
            return

        print("\n=== Data Visualization Options ===")
        print("1. Temperature Trends")
        print("2. Humidity vs Temperature")
        print("3. Wind Speed Distribution")
        print("4. Correlation Matrix")
        
        choice = input("\nSelect visualization type (1-4): ")
        
        plt.figure(figsize=(12, 6))
        
        try:
            if choice == '1':
                plt.plot(self.historical_data['date'], self.historical_data['max_temp'], label='Max Temp')
                plt.plot(self.historical_data['date'], self.historical_data['min_temp'], label='Min Temp')
                plt.title('Temperature Trends Over Time')
                plt.xlabel('Date')
                plt.ylabel('Temperature (°F)')
                plt.xticks(rotation=45)
                plt.legend()
            elif choice == '2':
                plt.scatter(self.historical_data['humidity'], self.historical_data['max_temp'], 
                          alpha=0.5, label='Max Temp')
                plt.scatter(self.historical_data['humidity'], self.historical_data['min_temp'], 
                          alpha=0.5, label='Min Temp')
                plt.title('Humidity vs Temperature')
                plt.xlabel('Humidity (%)')
                plt.ylabel('Temperature (°F)')
                plt.legend()
            elif choice == '3':
                sns.histplot(self.historical_data['wind_speed'], bins=20)
                plt.title('Wind Speed Distribution')
                plt.xlabel('Wind Speed (mph)')
            elif choice == '4':
                numeric_cols = self.historical_data.select_dtypes(include=[np.number]).columns
                correlation_matrix = self.historical_data[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Matrix of Weather Variables')
            else:
                print("Invalid choice!")
                return

            # Save and show the plot
            plt.tight_layout()
            save_path = os.path.join(
                self.data_dir, 
                'visualizations', 
                f'viz_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            )
            plt.savefig(save_path)
            plt.show()
            print(f"\nVisualization saved to: {save_path}")

        except Exception as e:
            print(f"Error creating visualization: {e}")

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