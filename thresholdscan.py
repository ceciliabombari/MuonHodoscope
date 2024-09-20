import struct
import numpy as np
import matplotlib.pyplot as plt  # Import the matplotlib library for plotting
import pandas as pd  # Import pandas for handling data in tabular form

# Define the file path to the binary data file
file_path = '/mnt/c/Users/cecil/Downloads/sensor-scan.147.dat'

# Define the word size (32 bits = 4 bytes)
word_size = 4  # 4 bytes per word (32 bits)
header_size = 4 * word_size  # The header consists of 4 words (16 bytes)
data_block_size = 56 * word_size  # The data block consists of 56 words (224 bytes)
block_size = header_size + data_block_size  # Total size of each block (header + data)

# Prepare a structure to store thresholds and sensor data
thresholds = []  # List to store the threshold values
sensor_data = [[] for _ in range(56)]  # List of lists to store data for 56 sensors

# Function to extract the threshold value from the 3rd word of the header
def extract_threshold(word):
    # Mask to keep only the upper 16 bits of the word (which contains the threshold)
    threshold = (word >> 16) & 0xFFFF
    return threshold

# Read the binary file and extract thresholds and sensor data
with open(file_path, 'rb') as file:
    while True:
        # Read each block of data (header + sensor data)
        block = file.read(block_size)
        
        if not block:
            break  # Exit the loop when the file ends
        
        # Unpack the header (4 words from the beginning of the block)
        header = struct.unpack('4I', block[:header_size])
        
        # Extract the threshold value from the 3rd word of the header
        threshold = extract_threshold(header[2])
        
        # Only process data if the threshold is within the desired range (1790 <= threshold <= 2100)
        if 1790 <= threshold <= 2100:
            thresholds.append(threshold)  # Append the threshold value to the list
            
            # Unpack the data part (56 words representing sensor data)
            data = struct.unpack('56I', block[header_size:])
            
            # Store the sensor data (excluding the inactive sensors)
            for i in range(56):
                sensor_data[i].append(data[i])

# Convert lists to NumPy arrays for easier data manipulation
thresholds = np.array(thresholds)
sensor_data = [np.array(sensor) for sensor in sensor_data]

# Plot the sensor data: Active sensors (1 to 26 and 29 to 52)
plt.figure(figsize=(12, 8))  # Create a figure for the plot

# Loop through the active sensors (1-26 and 29-52)
for i in range(26):
    if len(sensor_data[i]) > 0:  # Plot only if sensor data exists
        frequency = np.array(sensor_data[i]) / 10 / 1000  # Convert data to kHz
        plt.plot(thresholds, frequency, label=f'Sensor {i+1}')  # Plot each sensor's data

for i in range(28, 52):
    if len(sensor_data[i]) > 0:  # Plot only if sensor data exists
        frequency = np.array(sensor_data[i]) / 10 / 1000  # Convert data to kHz
        plt.plot(thresholds, frequency, label=f'Sensor {i+1}')

# Set y-axis to logarithmic scale to show a wide range of frequencies
plt.yscale('log')

# Add labels and a title to the plot
plt.xlabel('Threshold')
plt.ylabel('Firing Events Frequency (kHz)')
plt.title('THRESHOLD SCAN')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)  # Position the legend
plt.grid(True)  # Add gridlines to the plot
plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('threshold_scan_plot147.png')  # Save the plot to a file

# Plot for inactive sensors (27, 28, 53-56)
plt.figure(figsize=(12, 8))  # Create a new figure

# Loop through the inactive sensors
for i in [26, 27, 52, 53, 54, 55]:
    if len(sensor_data[i]) > 0:  # Plot only if sensor data exists
        frequency = np.array(sensor_data[i]) / 10 / 1000  # Convert data to kHz
        plt.plot(thresholds, frequency, label=f'Sensor {i+1}')  # Plot each sensor's data

# Set y-axis to logarithmic scale to capture the variation in noise levels
plt.yscale('log')

# Add labels and title to the plot
plt.xlabel('Threshold')
plt.ylabel('Inactive Sensors Noise (kHz)')
plt.title('THRESHOLD SCAN (Inactive Sensors)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)  # Position the legend
plt.grid(True)  # Add gridlines
plt.tight_layout()  # Adjust layout
plt.savefig('/home/ceci/Coding/threshold_noise.png')  # Save the plot to a file

# Find the minimum threshold for each sensor at 1 kHz frequency (and other frequencies in `j` loop)
for j in [1, 2, 10, 60, 100]:  # Frequencies to check
    min_thresholds = []  # List to store minimum thresholds for each sensor
    for i in range(56):
        if len(sensor_data[i]) > 0:
            frequency = np.array(sensor_data[i]) / 1000  # Convert sensor data to kHz
            try:
                # Find the first threshold where frequency is greater than or equal to `j` kHz
                index = np.where(frequency >= j)[0][0]
                min_threshold = thresholds[index]  # Extract the corresponding threshold
            except IndexError:
                # If no frequency >= `j`, assign 0 as the minimum threshold
                min_threshold = 0
            min_thresholds.append({'Sensor': i + 1, 'Min Threshold': min_threshold})  # Store result

    # Create a DataFrame to organize the minimum thresholds
    df = pd.DataFrame(min_thresholds)

    # Display the DataFrame with sensor number and minimum threshold
    print(df)

    # Define the filename for saving results as a text file
    txt_filename = f"threshold_scan_147_{j * 100}Hz.txt"

    # Save the DataFrame to a text file in a neatly formatted way
    with open(txt_filename, 'w') as f:
        # Write a header for clarity
        f.write(f"{'Sensor':<10}{'Min Threshold':<15}\n")
        f.write(f"{'-'*10}{'-'*15}\n")
        # Write each row of the DataFrame with proper spacing
        for _, row in df.iterrows():
            f.write(f"{row['Sensor']:<10}{row['Min Threshold']:<15}\n")

    print(f"Data saved to {txt_filename}")  # Notify the user that data has been saved

