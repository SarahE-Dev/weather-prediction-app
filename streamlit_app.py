import streamlit as st  
import time 
import os  
import matplotlib.pyplot as plt
from src.streamlit_weather_app import WeatherApp 
import numpy as np  
import pandas as pd  

def setup_dark_mode():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#1C1C1C',  
        'axes.facecolor': '#2D2D2D',    
        'axes.labelcolor': 'white',     
        'text.color': 'white',       
        'xtick.color': 'white',        
        'ytick.color': 'white',     
        'grid.color': '#444444',     
        'legend.facecolor': '#2D2D2D', 
        'legend.edgecolor': 'white'     
    })

setup_dark_mode()

def create_streamlit_app():
    """
    Create and run the Streamlit web application for the Weather Prediction App.
    
    This function sets up the Streamlit app, initializes the WeatherApp object,
    and provides an interactive interface for users to:
    - Collect historical weather data
    - Record new observations
    - View observation photos
    - Train and load prediction models
    - Get temperature predictions
    - Visualize weather data
    - Compare predictions with actual observations
    """
    # Set the page configuration for the Streamlit app
    st.set_page_config(
        page_title="Weather Prediction App",
        page_icon="🌤️",
        layout="wide"
    )

    # Initialize the WeatherApp object in the session state if not already present
    if 'weather_app' not in st.session_state:
        st.session_state.weather_app = WeatherApp()

    # Display the app title and a horizontal line
    st.title("Weather Prediction App")
    st.markdown("---")

    # Define the menu options for the sidebar
    menu_options = [
        "Collect Historical Data",
        "Load Existing Historical Data",
        "Record New Observation",
        "View Observation Photos",
        "Train New Model",
        "Load Existing Model",
        "Get Temperature Prediction",
        "Visualize Data",
        "Compare Predictions"
    ]

    # Create a sidebar menu for users to select an operation
    choice = st.sidebar.selectbox("Select Operation", menu_options)

    # Handle each menu option with appropriate functionality
    if choice == "Collect Historical Data":
        st.header("Collect Historical Weather Data")
        # Input fields for city name and number of days
        city = st.text_input("Enter city name:")
        days = st.number_input("Enter number of days:", min_value=1, max_value=180, value=7)
        
        # Button to initiate data collection
        if st.button("Collect Data"):
            with st.spinner("Collecting historical data..."):
                try:
                    # Use the WeatherApp instance to collect historical data
                    st.session_state.weather_app.collect_historical_data(city, days)
                    st.success("Data collected successfully!")
                    
                    # Display the data if available
                    if st.session_state.weather_app.historical_data is not None:
                        st.subheader("Historical Weather Data")
                        
                        # Display summary statistics using metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Average High", 
                                f"{st.session_state.weather_app.historical_data['max_temp'].mean():.1f}°F"
                            )
                        with col2:
                            st.metric(
                                "Average Low", 
                                f"{st.session_state.weather_app.historical_data['min_temp'].mean():.1f}°F"
                            )
                        with col3:
                            st.metric(
                                "Average Humidity", 
                                f"{st.session_state.weather_app.historical_data['humidity'].mean():.1f}%"
                            )
                        
                        # Create a temperature range plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(
                            st.session_state.weather_app.historical_data['date'], 
                            st.session_state.weather_app.historical_data['max_temp'], 
                            label='High', color='red'
                        )
                        ax.plot(
                            st.session_state.weather_app.historical_data['date'], 
                            st.session_state.weather_app.historical_data['min_temp'], 
                            label='Low', color='blue'
                        )
                        ax.fill_between(
                            st.session_state.weather_app.historical_data['date'],
                            st.session_state.weather_app.historical_data['max_temp'],
                            st.session_state.weather_app.historical_data['min_temp'],
                            alpha=0.2
                        )
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Temperature (°F)')
                        ax.set_title(f'Temperature Range for {city}')
                        ax.grid(True)
                        ax.legend()
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        # Display the plot in the Streamlit app
                        st.pyplot(fig)
                        
                        # Provide an option to view the raw data in a table
                        with st.expander("View Raw Data"):
                            st.dataframe(st.session_state.weather_app.historical_data)
                    
                except Exception as e:
                    # Display an error message if data collection fails
                    st.error(f"Error collecting data: {str(e)}")

    elif choice == "Load Existing Historical Data":
        st.header("Load Existing Historical Data")
        # Button to load existing data
        if st.button("Load Data"):
            with st.spinner("Loading historical data..."):
                if st.session_state.weather_app.load_historical_data():
                    st.success("Data loaded successfully!")
                    
                    # Display the data if available
                    if st.session_state.weather_app.historical_data is not None:
                        st.subheader("Historical Weather Data")
                        
                        # Display summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Average High", 
                                f"{st.session_state.weather_app.historical_data['max_temp'].mean():.1f}°F"
                            )
                        with col2:
                            st.metric(
                                "Average Low", 
                                f"{st.session_state.weather_app.historical_data['min_temp'].mean():.1f}°F"
                            )
                        with col3:
                            st.metric(
                                "Average Humidity", 
                                f"{st.session_state.weather_app.historical_data['humidity'].mean():.1f}%"
                            )
                        
                        # Create a temperature range plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(
                            st.session_state.weather_app.historical_data['date'], 
                            st.session_state.weather_app.historical_data['max_temp'], 
                            label='High', color='red'
                        )
                        ax.plot(
                            st.session_state.weather_app.historical_data['date'], 
                            st.session_state.weather_app.historical_data['min_temp'], 
                            label='Low', color='blue'
                        )
                        ax.fill_between(
                            st.session_state.weather_app.historical_data['date'],
                            st.session_state.weather_app.historical_data['max_temp'],
                            st.session_state.weather_app.historical_data['min_temp'],
                            alpha=0.2
                        )
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Temperature (°F)')
                        ax.set_title('Historical Temperature Range')
                        ax.grid(True)
                        ax.legend()
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Provide an option to view the raw data
                        with st.expander("View Raw Data"):
                            st.dataframe(st.session_state.weather_app.historical_data)
                else:
                    # Display an error if data loading fails
                    st.error("No existing data found or error loading data.")

    elif choice == "Record New Observation":
        st.header("Record New Weather Observation")
        
        # Create two columns for input fields
        col1, col2 = st.columns(2)
        
        with col1:
            # Input fields for weather parameters
            temperature = st.number_input(
                "Temperature (°F):", 
                min_value=-30.0, max_value=150.0, value=70.0, 
                step=1.0
            )
            
            humidity = st.number_input(
                "Humidity (%):", 
                min_value=0.0, max_value=100.0, 
                step=1.0
            )
            
            cloud_cover = st.selectbox(
                "Cloud Cover:",
                options=["clear", "partly", "cloudy"]
            )
            
            wind_speed = st.number_input(
                "Wind Speed (mph):", 
                min_value=0.0, max_value=200.0, 
                step=1.0
            )
            
            wind_direction = st.selectbox(
                "Wind Direction:",
                options=[
                    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
                ]
            )
        
        with col2:
            precipitation = st.selectbox(
                "Precipitation:",
                options=["no", "yes"]
            )
            
            pressure = st.number_input(
                "Pressure (inHg):", 
                min_value=20.0, max_value=32.0, value=29.92,
                step=0.01, format="%.2f"
            )
            
            visibility = st.number_input(
                "Visibility (miles):", 
                min_value=0.0, max_value=100.0, 
                step=0.1
            )
            
            notes = st.text_area("Notes:", height=100)
        
        # File uploader for the weather photo
        photo = st.file_uploader("Upload Weather Photo", type=['jpg', 'jpeg', 'png'])
        
        # Button to record the observation
        if st.button("Record Observation"):
            with st.spinner("Recording observation..."):
                try:
                    # Prepare the observation data
                    observation_data = {
                        'temperature': temperature,
                        'humidity': humidity,
                        'cloud_cover': cloud_cover,
                        'wind_speed': wind_speed,
                        'wind_direction': wind_direction,
                        'precipitation': precipitation,
                        'pressure': pressure,
                        'visibility': visibility,
                        'notes': notes
                    }
                    # Use the WeatherApp instance to record the observation
                    result = st.session_state.weather_app.record_observation(observation_data, photo)
                    st.success("Observation recorded successfully!")
                    st.write("Recorded Observation:", result)
                except Exception as e:
                    # Display an error message if recording fails
                    st.error(f"Error recording observation: {str(e)}")

    elif choice == "View Observation Photos":
        st.header("View Observation Photos")
        # Button to load photos
        if st.button("Load Photos"):
            with st.spinner("Loading photos..."):
                try:
                    # Retrieve the photos from the WeatherApp instance
                    photos = st.session_state.weather_app.view_observation_photos()
                    if photos:
                        # Display each photo with its filename
                        for photo in photos:
                            st.write(f"Filename: {photo['filename']}")
                            st.image(photo['image'])
                            st.markdown("---")
                    else:
                        st.info("No photos found in observations/photos directory.")
                except Exception as e:
                    # Display an error message if photo loading fails
                    st.error(f"Error loading photos: {str(e)}")

    elif choice == "Train New Model":
        st.header("Train New Weather Prediction Model")
        # Button to initiate model training
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    # Train the model using the WeatherApp instance
                    metrics = st.session_state.weather_app.train_model()
                    
                    # Display model performance metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label="Mean Absolute Error (MAE)", 
                            value=f"{metrics['mae']:.2f}°F",
                            help="Average absolute difference between predicted and actual temperatures"
                        )
                    
                    with col2:
                        st.metric(
                            label="Root Mean Square Error (RMSE)", 
                            value=f"{metrics['rmse']:.2f}°F",
                            help="Root mean square error of temperature predictions"
                        )
                    
                    with st.expander("Features Used in Predictions"):
                        st.write("The model uses these weather conditions to make predictions:")
                        st.write("• Humidity")
                        st.write("• Precipitation")
                        st.write("• Wind Speed")
                        st.write("• Average Temperature")
                        st.write("• Time of Year")
                    
                    st.success("Model trained successfully!")
                except Exception as e:
                    # Display an error message if training fails
                    st.error(f"Error training model: {str(e)}")

    elif choice == "Load Existing Model":
        st.header("Load Existing Model")
        # Button to load the model
        if st.button("Load Model"):
            with st.spinner("Loading model..."):
                try:
                    # Load the model using the WeatherApp instance
                    success = st.session_state.weather_app.load_model()
                    if success:
                        # Retrieve metrics from the loaded model
                        metrics = {
                            'mae': st.session_state.weather_app.mae,
                            'rmse': st.session_state.weather_app.rmse
                        }
                        
                        # Display the model performance metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                label="Mean Absolute Error (MAE)", 
                                value=f"{metrics['mae']:.2f}°F",
                                help="Average absolute difference between predicted and actual temperatures"
                            )
                        
                        with col2:
                            st.metric(
                                label="Root Mean Square Error (RMSE)", 
                                value=f"{metrics['rmse']:.2f}°F",
                                help="Root mean square error of temperature predictions"
                            )
                            
                        st.success("Model loaded successfully!")
                        
                        with st.expander("Features Used in Predictions"):
                            st.write("The model uses these weather conditions to make predictions:")
                            st.write("• Humidity")
                            st.write("• Precipitation")
                            st.write("• Wind Speed")
                            st.write("• Average Temperature")
                            st.write("• Time of Year")
                    else:
                        st.error("No existing model found or error loading model.")
                except Exception as e:
                    # Display an error message if loading fails
                    st.error(f"Error loading model: {str(e)}")

    elif choice == "Get Temperature Prediction":
        st.header("Temperature Prediction")
        # Button to get the prediction
        if st.button("Get Prediction"):
            with st.spinner("Generating prediction..."):
                try:
                    # Generate the temperature prediction using the WeatherApp instance
                    prediction_data = st.session_state.weather_app.predict_temperature()
                    st.title("Weather Prediction Results")

                    # Extract predicted high and low temperatures
                    predicted_high = prediction_data["prediction"]["predicted_high"]
                    predicted_low = prediction_data["prediction"]["predicted_low"]
                    prediction_date = prediction_data["prediction"]["date"]

                    # Display the prediction results
                    st.subheader(f"Prediction for {prediction_date}")
                    st.metric(label="Predicted High (°F)", value=f"{predicted_high:.2f}")
                    st.metric(label="Predicted Low (°F)", value=f"{predicted_low:.2f}")
                except Exception as e:
                    # Display an error message if prediction fails
                    st.error(f"Error generating prediction: {str(e)}")

    elif choice == "Visualize Data":
        st.header("Data Visualization")
        
        # Check if historical data is available
        if st.session_state.weather_app.historical_data is None:
            st.warning("Please load historical data first!")
        else:
            # Select the type of visualization
            viz_type = st.selectbox(
                "Select visualization type:",
                [
                    "Temperature Trends", 
                    "Humidity vs Temperature", 
                    "Wind Speed Distribution", 
                    "Correlation Matrix", 
                    "Temperature Heatmap"
                ]
            )
            
            # Button to generate the visualization
            if st.button("Generate Visualization"):
                with st.spinner("Creating visualization..."):
                    try:
                        # Generate the visualization using the WeatherApp instance
                        fig = st.session_state.weather_app.visualize_data(viz_type)
                        st.pyplot(fig)
                        st.success("Visualization created successfully!")
                    except Exception as e:
                        # Display an error message if visualization fails
                        st.error(f"Error creating visualization: {str(e)}")
    
    elif choice == "Compare Predictions":
        st.header("Compare Predictions with Observations")
        
        # Load predictions and observations for comparison
        try:
            # Define paths to the prediction and observation files
            predictions_file = os.path.join('data', 'predictions', 'predictions.csv')
            observations_file = os.path.join('data', 'observations', 'observations.csv')
            
            # Check if both files exist
            if not os.path.exists(predictions_file) or not os.path.exists(observations_file):
                st.warning("Both predictions and observations files must exist to make comparisons.")
            else:
                # Read the CSV files into DataFrames
                predictions_df = pd.read_csv(predictions_file)
                observations_df = pd.read_csv(observations_file)
                
                # Convert 'date' columns to datetime objects
                predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.date
                observations_df['date'] = pd.to_datetime(observations_df['date']).dt.date
                
                # Find dates present in both predictions and observations
                matching_dates = set(predictions_df['date']) & set(observations_df['date'])
                
                if not matching_dates:
                    st.warning("No matching dates found between predictions and observations.")
                else:
                    st.success(f"Found {len(matching_dates)} dates with both predictions and observations.")
                    
                    # Prepare comparison data
                    comparison_data = []
                    for date in matching_dates:
                        # Get prediction for the date
                        pred = predictions_df[predictions_df['date'] == date].iloc[0]
                        # Get all observations for the date
                        obs = observations_df[observations_df['date'] == date]
                        
                        # Calculate actual high and low temperatures from observations
                        actual_high = obs['temperature'].max()
                        actual_low = obs['temperature'].min()
                        
                        # Append comparison data
                        comparison_data.append({
                            'date': date,
                            'predicted_high': pred['predicted_high'],
                            'actual_high': actual_high,
                            'high_error': pred['predicted_high'] - actual_high,
                            'predicted_low': pred['predicted_low'],
                            'actual_low': actual_low,
                            'low_error': pred['predicted_low'] - actual_low
                        })
                    
                    # Create a DataFrame from the comparison data
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Display summary error metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        mae_high = abs(comparison_df['high_error']).mean()
                        st.metric("High Temperature MAE", f"{mae_high:.1f}°F")
                    with col2:
                        mae_low = abs(comparison_df['low_error']).mean()
                        st.metric("Low Temperature MAE", f"{mae_low:.1f}°F")
                    
                    # Create comparison plots
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                    
                    # Convert dates back to datetime for plotting
                    comparison_df['date'] = pd.to_datetime(comparison_df['date'])
                    
                    # Plot predicted vs actual high temperatures
                    ax1.plot(
                        comparison_df['date'], comparison_df['predicted_high'], 
                        marker='o', label='Predicted High', color='red'
                    )
                    ax1.plot(
                        comparison_df['date'], comparison_df['actual_high'], 
                        marker='o', label='Actual High', color='darkred', linestyle='--'
                    )
                    ax1.set_title('High Temperature Comparison')
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel('Temperature (°F)')
                    ax1.legend()
                    ax1.grid(True)
                    
                    # Plot predicted vs actual low temperatures
                    ax2.plot(
                        comparison_df['date'], comparison_df['predicted_low'], 
                        marker='o', label='Predicted Low', color='blue'
                    )
                    ax2.plot(
                        comparison_df['date'], comparison_df['actual_low'], 
                        marker='o', label='Actual Low', color='darkblue', linestyle='--'
                    )
                    ax2.set_title('Low Temperature Comparison')
                    ax2.set_xlabel('Date')
                    ax2.set_ylabel('Temperature (°F)')
                    ax2.legend()
                    ax2.grid(True)
                    
                    plt.tight_layout()
                    # Display the comparison plots
                    st.pyplot(fig)
                    
                    # Display the detailed comparison table
                    st.subheader("Detailed Comparison")
                    comparison_display = comparison_df.copy()
                    comparison_display['date'] = comparison_display['date'].dt.strftime('%Y-%m-%d')
                    # Round numerical values for display
                    st.dataframe(comparison_display.round(1))
        except Exception as e:
            # Display an error message if comparison fails
            st.error(f"Error comparing predictions: {str(e)}")

if __name__ == "__main__":
    # Run the Streamlit app
    create_streamlit_app()