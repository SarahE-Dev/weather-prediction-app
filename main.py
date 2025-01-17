# main.py

from src.weather_app import WeatherApp
import sys
import time
import os

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_banner():
    """Display the application banner."""
    banner = """
    ╔══════════════════════════════════════╗
    ║        Weather Prediction App        ║
    ╚══════════════════════════════════════╝
    """
    print(banner)

def display_menu():
    """Display the main menu options."""
    menu = """
    1. Collect Historical Data
    2. Load Existing Historical Data
    3. Record New Observation
    4. View Observation Photos
    5. Train New Model
    6. Load Existing Model
    7. Get Temperature Prediction
    8. Visualize Data
    9. Export Data
    10. Exit
    """
    print(menu)

def main():
    """Main application entry point."""
    app = WeatherApp()
    
    try:
        while True:
            clear_screen()
            display_banner()
            display_menu()
            
            choice = input("\nEnter your choice (1-10): ").strip()
            
            if choice == '1':
                clear_screen()
                print("=== Collect Historical Weather Data ===")
                app.collect_historical_data()
                input("\nPress Enter to continue...")

            elif choice == '2':
                clear_screen()
                print("=== Load Existing Historical Data ===")
                if app.load_historical_data():
                    print("\nData loaded successfully!")
                else:
                    print("\nNo existing data found or error loading data.")
                input("\nPress Enter to continue...")

            elif choice == '3':
                clear_screen()
                print("=== Record New Weather Observation ===")
                app.record_observation()
                input("\nPress Enter to continue...")
                
            elif choice == '4':
                clear_screen()
                print("=== View Observation Photos ===")
                app.view_observation_photos()
                input("\nPress Enter to continue...")

            elif choice == '5':
                clear_screen()
                print("=== Train New Weather Prediction Model ===")
                app.train_model()  # Call the train_model method of WeatherApp
                input("\nPress Enter to continue...")

            elif choice == '6':
                clear_screen()
                print("=== Load Existing Model ===")
                if app.load_model():
                    print("\nModel loaded successfully!")
                else:
                    print("\nNo existing model found or error loading model.")
                input("\nPress Enter to continue...")

            elif choice == '7':
                clear_screen()
                print("=== Temperature Prediction ===")
                app.predict_temperature()
                input("\nPress Enter to continue...")

            elif choice == '8':
                clear_screen()
                print("=== Data Visualization ===")
                app.visualize_data()
                input("\nPress Enter to continue...")

            elif choice == '9':
                clear_screen()
                print("=== Export Data ===")
                app.export_data()
                input("\nPress Enter to continue...")

            elif choice == '10':
                clear_screen()
                print("\nThank you for using the Weather Prediction App!")
                print("Saving data and cleaning up...")
                time.sleep(1)
                print("Goodbye!")
                break

            else:
                print("\nInvalid choice. Please try again.")
                time.sleep(1)

    except KeyboardInterrupt:
        clear_screen()
        print("\n\nExiting gracefully. Goodbye!")
    except Exception as e:
        clear_screen()
        print(f"\nAn unexpected error occurred: {str(e)}")
        print("The application will now exit.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)