# src/record_observation.py
import pandas as pd
from datetime import datetime
from PIL import Image

def record_observation():
    date = datetime.now().strftime('%Y-%m-%d')
    time = datetime.now().strftime('%H:%M')
    
    print("\n=== Weather Observation Recording ===")
    temp = float(input("Temperature (°F): "))
    
    print("\nCloud Cover Options: clear, partly, cloudy")
    cloud_cover = input("Cloud Cover: ")
    
    print("\nWind Options: calm, breezy, windy")
    wind = input("Wind Condition: ")
    
    print("\nPrecipitation (yes/no)")
    precip = input("Precipitation: ")
    
    photo_filename = f"photo_{date}_{time.replace(':', '')}.jpg"
    print(f"\nPlease save your sky photo as: {photo_filename}")
    
    df = pd.read_csv('data/observations/observations.csv')
    new_observation = {
        'date': date,
        'time': time,
        'temperature': temp,
        'cloud_cover': cloud_cover,
        'wind': wind,
        'precipitation': precip,
        'photo_filename': photo_filename
    }
    df = df.append(new_observation, ignore_index=True)
    df.to_csv('data/observations/observations.csv', index=False)
    print("\nObservation recorded successfully!")