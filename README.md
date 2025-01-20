# Weather Prediction App

Welcome to the Weather Prediction App! This application allows you to collect historical weather data, train a predictive model, record observations, make predictions, and visualize data. The app is built using Python and leverages various libraries for data processing, modeling, and visualization.

## Features

- Collect historical weather data for a specified city and number of days.
- Train a predictive model using the collected data.
- Record and store weather observations.
- Make weather predictions and evaluate model performance.
- Visualize weather data and prediction results.

## Installation

To get started with the Weather Prediction App, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd weather-prediction-app
Install the required dependencies:

Make sure you have Python installed. Then, install the necessary packages using pip:

BASH

pip install -r requirements.txt
Set up directories:

The app will automatically create necessary directories for storing data, models, and visualizations.

Usage
To run the Weather Prediction App, use the following command:

BASH

streamlit run streamlit_app.py
This will start the Streamlit server and open the app in your default web browser.

Code Overview
Main Components
WeatherApp Class: The core class representing the weather application. It handles data collection, model training, predictions, and visualizations.

Data Collection: Uses the get_historical_weather function to fetch historical weather data from an API.

Data Processing: Processes raw data using the process_data function, including cleaning and feature engineering.

Model Building: The build_model function is used to train a predictive model on the processed data.


Dependencies
The app uses the following Python libraries:

os: For directory management.
pandas: For data manipulation and analysis.
numpy: For numerical operations.
datetime: For handling date and time operations.
matplotlib: For data visualization.
seaborn: For statistical data visualization.
sklearn: For machine learning model evaluation.
joblib: For model serialization and deserialization.