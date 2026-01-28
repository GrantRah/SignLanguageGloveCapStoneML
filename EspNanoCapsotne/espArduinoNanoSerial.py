import serial
import time
import csv

# === CONFIG ===
PORT = "COM5"       # Update with your ESP32 port
BAUD_RATE = 115200
OUTPUT_FILE = "gesture_dataahh.csv"

# Connect to ESP32
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # wait for ESP32 to reset

print("Connected to ESP32")

# Storage for 2D array
data = []

# Temporary variables to store latest acceleration values
ax = ay = az = None
gesture = None

try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        if not line:
            continue

        # Skip startup messages
        if line.startswith("Serial connection ready") or line.startswith("MPU6050 Found!"):
            continue

        # ---- Parse Gesture ----
        if line.startswith("GESTURE"):
            gesture = int(line.split("0")[-1].strip())  # gets '1' or '2'
            print(f"Detected Gesture: {gesture}")

        # ---- Parse Acceleration ----
        elif line.startswith("Acceleration"):
            # Example: Acceleration: X:0.12 Y:1.23 Z:9.81
            parts = line.replace("Acceleration:", "").split()
            ax = float(parts[0].split(":")[1])
            ay = float(parts[1].split(":")[1])
            az = float(parts[2].split(":")[1])
            print(f"Ax={ax:.2f}, Ay={ay:.2f}, Az={az:.2f}")

            # Only save row if we already have a gesture value
            if gesture is not None:
                row = [ax, ay, az, gesture]
                data.append(row)
                gesture = None  # reset gesture for next row

except KeyboardInterrupt:
    print("\nStopping serial read...")
finally:
    # Write CSV
    with open(OUTPUT_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Ax", "Ay", "Az", "Gesture"])  # header
        writer.writerows(data)

    print(f"Data saved to {OUTPUT_FILE}")
