import os
import serial
import pandas as pd

# Serial port configuration
arduino_port = 'COM9'  # Replace with your Arduino's COM port
baud_rate = 9600

# Specify the folder path
folder_path = r"C:\Users\yugde\OneDrive\Desktop\Tactile_sensor_data"
# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# Full path for the output file
output_file = os.path.join(folder_path, "sensor_data_grid_(3,5).csv")

# Column names for the dataset
columns = [
    "Force (N)", 
    "Flex Raw ADC 1", 
    "Flex Voltage 1(V)", 
    "Flex Resistance 1(Ohms)",
    "Flex Raw ADC 2", 
    "Flex Voltage 2(V)", 
    "Flex Resistance 2(Ohms)",
    "Flex Raw ADC 3", 
    "Flex Voltage 3(V)", 
    "Flex Resistance 3(Ohms)", 
    "Load Cell Voltage (V)", 
    "Load Cell Raw ADC"
]

# Initialize serial connection
try:
    arduino = serial.Serial(arduino_port, baud_rate, timeout=2)
    print(f"Connected to Arduino on {arduino_port} at {baud_rate} baud.")
except serial.SerialException as e:
    print(f"Error: Could not open port {arduino_port}. Details: {e}")
    exit()

# Data storage
data = []

try:
    print("Collecting data... Press Ctrl+C to stop.")
    while True:
        # Read a line of data from Arduino
        line = arduino.readline()
        try:
            # Decode the line with error handling
            decoded_line = line.decode(errors='replace').strip()
            if decoded_line:
                print(decoded_line)  # Print to console for live monitoring
                
                # Skip the header row if it's sent again
                if any(header in decoded_line for header in columns):
                    continue
                
                # Split and convert the data into floats
                values = decoded_line.split(",")
                if len(values) == len(columns):  # Ensure data matches expected format
                    try:
                        data.append([float(v) for v in values])  # Convert to float and store
                    except ValueError:
                        print(f"Warning: Skipped invalid data row: {decoded_line}")
        except Exception as e:
            print(f"Error decoding line: {e}")
except KeyboardInterrupt:
    print("\nData collection stopped by user.")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    # Close the serial connection and save the data
    print("Closing serial connection...")
    arduino.close()
    if data:
        # Save collected data to a CSV file
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    else:
        print("No data collected. Exiting.")
