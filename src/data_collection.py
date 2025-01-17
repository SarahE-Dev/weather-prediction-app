import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from .config import API_KEY, CITY

def get_historical_weather(city, days):
    """
    Collect historical weather data using WeatherAPI.
    
    Args:
        city (str): City name or location
        days (int): Number of days of historical data to collect
        
    Returns:
        pandas.DataFrame: Historical weather data
    """
    # Create directories if they don't exist
    os.makedirs('data/historical', exist_ok=True)
    os.makedirs('data/observations/photos', exist_ok=True)

    base_url = "http://api.weatherapi.com/v1/history.json"
    weather_data = []
    end_date = datetime.now()
    
    print(f"\nCollecting historical weather data for {city}...")
    print(f"Collecting {days} days of data...")
    
    # Collect days of historical data
    for i in range(days):
        date = end_date - timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')
        
        params = {
            'key': API_KEY,
            'q': city,
            'dt': date_str
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract daily weather data
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
            weather_data.append(daily_data)
            print(f"✓ Collected data for {date_str}")
            
        except requests.exceptions.RequestException as e:
            print(f"× Error collecting data for {date_str}: {str(e)}")
            continue
        except KeyError as e:
            print(f"× Error parsing data for {date_str}: {str(e)}")
            continue
        except Exception as e:
            print(f"× Unexpected error for {date_str}: {str(e)}")
            continue
    
    if not weather_data:
        raise Exception("No weather data was collected successfully")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(weather_data)
    
    # Convert precipitation from mm to binary (0 for no precipitation, 1 for precipitation)
    df['precipitation'] = (df['precipitation'] > 0).astype(int)
    
    # Convert wind speed from kph to mph
    df['wind_speed'] = df['wind_speed'] * 0.621371
    
    # Save the data
    output_file = 'data/historical/weather_history.csv'
    df.to_csv(output_file, index=False)
    print(f"\nData saved to {output_file}")
    print(f"Successfully collected {len(df)} days of weather data")
    
    return df

def create_observation_template():
    """
    Create a template for manual weather observations.
    """
    # Create directories if they don't exist
    os.makedirs('data/observations', exist_ok=True)
    os.makedirs('data/observations/photos', exist_ok=True)
    
    # Define the structure for weather observations
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
    
    # Create empty DataFrame with defined columns
    df = pd.DataFrame(columns=columns)
    
    # Save empty template
    output_file = 'data/observations/observations.csv'
    df.to_csv(output_file, index=False)
    print(f"Created observation template at {output_file}")
    
    return df