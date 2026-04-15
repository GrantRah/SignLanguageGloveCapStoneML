"Receives serial data from ESP32, parses acceleration and gesture info, and saves to CSV."
"Once done collecting data, press Ctrl+C to stop and it will save automaticly to the CSV file."
import serial
import time
import csv

# =========================
# CONFIG
# =========================
PORT = "COM5"
BAUD_RATE = 115200
OUTPUT_FILE = "shape_data.csv"

# =========================
# CONNECT
# =========================
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
time.sleep(2)

print("Connected to ESP32. Logging data...")

data = []

# =========================
# LOOP
# =========================
try:
    while True:
        line = ser.readline().decode('utf-8').strip()

        if not line:
            continue

        # skip garbage/system messages
        if line.startswith("READY") or line.startswith("MPU"):
            continue

        parts = line.split(',')

        # expected format: gestureID, ax, ay, az
        if len(parts) != 4:
            continue

        try:
            gid = int(parts[0])
            ax = float(parts[1])
            ay = float(parts[2])
            az = float(parts[3])
        except:
            continue

        print(f"G{gid} | Ax={ax:.2f}, Ay={ay:.2f}, Az={az:.2f}")

        data.append([gid, ax, ay, az])

# =========================
# EXIT + SAVE
# =========================
except KeyboardInterrupt:
    print("\nStopping and saving...")

finally:
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["gesture_id", "Ax", "Ay", "Az"])
        writer.writerows(data)

    print(f"Saved to {OUTPUT_FILE}")