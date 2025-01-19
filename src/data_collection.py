import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from .config import API_KEY, CITY

def get_historical_weather(city, days):
    """
    Collect historical weather data using the WeatherAPI.

    Args:
        city (str): The name of the city or location to retrieve weather data for.
        days (int): The number of past days to collect historical data (up to 7 days per API call).

    Returns:
        pandas.DataFrame: A DataFrame containing the historical weather data.

    Raises:
        Exception: If no weather data was collected successfully.
    """
    # Create directories for storing data if they don't exist
    os.makedirs('data/historical', exist_ok=True)
    os.makedirs('data/observations/photos', exist_ok=True)

    # Base URL for the WeatherAPI historical data endpoint
    base_url = "http://api.weatherapi.com/v1/history.json"

    weather_data = []  # List to store weather data for each day
    end_date = datetime.now()

    # Collect weather data for the specified number of past days
    for i in range(days):
        # Calculate the date for the current iteration
        date = end_date - timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')

        # Set up the parameters for the API request
        params = {
            'key': API_KEY,
            'q': city,
            'dt': date_str
        }

        try:
            # Make the API request
            response = requests.get(base_url, params=params)
            # Raise an exception if the request was unsuccessful
            response.raise_for_status()
            # Parse the JSON data returned by the API
            data = response.json()

            # Extract daily weather data from the API response
            daily_data = {
                'date': date_str,
                'max_temp': data['forecast']['forecastday'][0]['day']['maxtemp_f'],
                'min_temp': data['forecast']['forecastday'][0]['day']['mintemp_f'],
                'avg_temp': data['forecast']['forecastday'][0]['day']['avgtemp_f'],
                'humidity': data['forecast']['forecastday'][0]['day']['avghumidity'],
                'precipitation': data['forecast']['forecastday'][0]['day']['totalprecip_mm'],
                'wind_speed': data['forecast']['forecastday'][0]['day']['maxwind_kph'],
                # Additional fields that might be useful
                'condition': data['forecast']['forecastday'][0]['day']['condition']['text'],
                'pressure': data['forecast']['forecastday'][0]['day'].get('pressure_mb', None),
                'visibility': data['forecast']['forecastday'][0]['day'].get('avgvis_km', None),
                'uv': data['forecast']['forecastday'][0]['day'].get('uv', None)
            }

            # Append the daily data to the list
            weather_data.append(daily_data)

        except requests.exceptions.RequestException as e:
            # Handle any exceptions related to the HTTP request
            print(f"Request exception for date {date_str}: {e}")
            continue  # Skip to the next day
        except KeyError as e:
            # Handle missing data in the API response
            print(f"Key error for date {date_str}: {e}")
            continue
        except Exception as e:
            # Handle any other exceptions
            print(f"Exception for date {date_str}: {e}")
            continue

    # Check if any weather data was collected
    if not weather_data:
        raise Exception("No weather data was collected successfully")

    # Create a DataFrame from the collected weather data
    df = pd.DataFrame(weather_data)

    # Data processing

    # Convert precipitation from millimeters to a binary indicator (0 for no precipitation, 1 for precipitation)
    df['precipitation'] = (df['precipitation'] > 0).astype(int)

    # Convert wind speed from kilometers per hour to miles per hour
    df['wind_speed'] = df['wind_speed'] * 0.621371

    # Save the processed data to a CSV file
    output_file = 'data/historical/weather_history.csv'
    df.to_csv(output_file, index=False)

    return df  # Return the DataFrame containing historical weather data

def create_observation_template():
    """
    Create a template CSV file for manual weather observations.

    The template includes the necessary columns for collecting weather observations,
    and saves an empty CSV file to the data/observations directory.

    Returns:
        pandas.DataFrame: An empty DataFrame with the observation columns.
    """
    # Create directories for observations if they don't exist
    os.makedirs('data/observations', exist_ok=True)
    os.makedirs('data/observations/photos', exist_ok=True)

    # Define the columns for the observation template
    columns = [
        'date',
        'time',
        'temperature',
        'humidity',
        'cloud_cover',
        'wind_speed',
        'wind_direction',
        'precipitation',
        'pressure',
        'visibility',
        'condition',
        'notes',
        'photo_filename'
    ]

    # Create an empty DataFrame with the defined columns
    df = pd.DataFrame(columns=columns)

    # Save the empty DataFrame as a CSV file
    output_file = 'data/observations/observations.csv'
    df.to_csv(output_file, index=False)

    return df  # Return the empty DataFrame