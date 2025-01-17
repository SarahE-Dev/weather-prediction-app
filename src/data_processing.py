# src/process_data.py
import pandas as pd
import numpy as np
from datetime import datetime

def process_data(data):
    """
    Process and clean the weather data.
    
    Args:
        data (pd.DataFrame): Raw weather data
        
    Returns:
        pd.DataFrame: Processed weather data
    """
    try:
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Convert date to datetime if it's not already
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        # Handle missing values in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # Replace missing values with column mean
            df[col] = df[col].fillna(df[col].mean())
            
            # Replace infinite values with column mean
            df[col] = df[col].replace([np.inf, -np.inf], df[col].mean())
            
            # Ensure all numeric data is float
            df[col] = df[col].astype(float)
        
        # Process specific columns
        if 'precipitation' in df.columns:
            # Convert precipitation to binary (0 for no rain, 1 for rain)
            df['precipitation'] = (df['precipitation'] > 0).astype(int)
            
        if 'wind_speed' in df.columns:
            # Ensure wind speed is non-negative
            df['wind_speed'] = df['wind_speed'].abs()
            
        if 'humidity' in df.columns:
            # Ensure humidity is between 0 and 100
            df['humidity'] = df['humidity'].clip(0, 100)
            
        if 'temperature' in df.columns:
            # Remove extreme temperature values (e.g., below -100°F or above 150°F)
            df['temperature'] = df['temperature'].clip(-100, 150)
            
        if 'max_temp' in df.columns:
            df['max_temp'] = df['max_temp'].clip(-100, 150)
            
        if 'min_temp' in df.columns:
            df['min_temp'] = df['min_temp'].clip(-100, 150)
        
        # Handle categorical columns
        if 'cloud_cover' in df.columns:
            # Standardize cloud cover categories
            df['cloud_cover'] = df['cloud_cover'].str.lower()
            valid_categories = ['clear', 'partly', 'cloudy']
            df['cloud_cover'] = df['cloud_cover'].apply(
                lambda x: x if x in valid_categories else 'partly'
            )
            
        if 'wind_direction' in df.columns:
            # Standardize wind direction
            df['wind_direction'] = df['wind_direction'].str.upper()
            valid_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            df['wind_direction'] = df['wind_direction'].apply(
                lambda x: x if x in valid_directions else None
            )
        
        # Add derived features
        if all(col in df.columns for col in ['max_temp', 'min_temp']):
            # Calculate temperature range
            df['temp_range'] = df['max_temp'] - df['min_temp']
        
        # Add time-based features if date is present
        if 'date' in df.columns:
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['season'] = df['date'].dt.month.apply(get_season)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Sort by date if present
        if 'date' in df.columns:
            df = df.sort_values('date')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        print("Data processing completed successfully!")
        print(f"Processed {len(df)} records")
        
        return df
    
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        # Return original data if processing fails
        return data

def get_season(month):
    """
    Convert month number to season.
    
    Args:
        month (int): Month number (1-12)
        
    Returns:
        str: Season name
    """
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'

def validate_data(data):
    """
    Validate the processed data meets requirements.
    
    Args:
        data (pd.DataFrame): Processed weather data
        
    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    try:
        # Check if data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            return False, "Data must be a pandas DataFrame"
        
        # Check if data is empty
        if len(data) == 0:
            return False, "Data is empty"
        
        # Required columns for model training
        required_columns = ['humidity', 'precipitation', 'wind_speed', 'max_temp', 'min_temp']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        # Check for missing values in required columns
        missing_values = data[required_columns].isnull().sum()
        if missing_values.any():
            return False, f"Missing values found in columns: {missing_values[missing_values > 0]}"
        
        # Check data types
        numeric_columns = required_columns
        for col in numeric_columns:
            if not np.issubdtype(data[col].dtype, np.number):
                return False, f"Column {col} must be numeric"
        
        return True, "Data validation successful"
    
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def prepare_model_data(data):
    """
    Prepare data specifically for model training.
    
    Args:
        data (pd.DataFrame): Processed weather data
        
    Returns:
        pd.DataFrame: Data ready for model training
    """
    try:
        # Select only the columns needed for modeling
        model_columns = ['humidity', 'precipitation', 'wind_speed', 'max_temp', 'min_temp']
        model_data = data[model_columns].copy()
        
        # Additional preprocessing specific to model requirements
        # Scale numeric features if needed
        # Handle any remaining issues
        
        return model_data
    
    except Exception as e:
        print(f"Error preparing model data: {str(e)}")
        return data

