import serial
import time
import csv
from datetime import datetime

# Configure serial port (change COM3 to your Pico's port)
ser = serial.Serial('COM3', 115200, timeout=1)
time.sleep(2)  # Wait for connection

# Create CSV file with timestamp
filename = f"flex_sensor_data_.csv"
csvfile = open(filename, 'w', newline='')
csvwriter = csv.writer(csvfile)

# Write header
csvwriter.writerow(['Sensor1(Index)', 'Sensor2(Middle)', 'Sensor3(Ring)', 'Constant'])

print(f"Logging to {filename}")
print("Press the button on Pico to start recording...")

recording = False
reading_count = 0
total_readings = 100

try:
    while True:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            
            # Check for recording start
            if "Recording Started" in line:
                recording = True
                reading_count = 0
                print("Recording started...")
                continue
            
            # Check for recording complete
            if "Recording Complete" in line:
                recording = False
                print("Recording complete!")
                continue
            
            # If recording, parse and save data
            if recording and line and line[0].isdigit():
                try:
                    values = line.split(',')
                    if len(values) == 4:  # Ensure we have 4 columns
                        csvwriter.writerow(values)
                        reading_count += 1
                        print(f"Saved reading {reading_count}/{total_readings}", end='\r')
                except:
                    pass

except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    csvfile.close()
    ser.close()
    print(f"\nData saved to {filename}")