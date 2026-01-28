import serial
import time
import numpy as np
import joblib

# === CONFIG ===
PORT = "COM5"
BAUD_RATE = 115200

# Load trained model and feature names
model = joblib.load("Trained_Nano_model.joblib")
feature_names = joblib.load("feature_Nano.joblib")

print("✓ Model loaded")
print(f"✓ Features: {feature_names}\n")

# Connect to Nano
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
time.sleep(2)

print("Connected to Nano. Starting live gesture prediction...\n")

while True:
    try:
        line = ser.readline().decode('utf-8').strip()
        if not line:
            continue

        # Skip startup messages
        if line.startswith("Serial connection ready") or line.startswith("MPU6050 Found!"):
            continue

        # Parse Acceleration line
        if line.startswith("Acceleration"):
            parts = line.replace("Acceleration:", "").split()
            try:
                ax = float(parts[0].split(":")[1])
                ay = float(parts[1].split(":")[1])
                az = float(parts[2].split(":")[1])
            except:
                continue

            # Create feature array
            X_input = np.array([[ax, ay, az]])

            # --- Prediction ---
            prediction = model.predict(X_input)[0]
            probabilities = model.predict_proba(X_input)[0]

            # --- Confidence analysis ---
            max_prob = np.max(probabilities) * 100

            if max_prob > 70:
                confidence = "High"
            elif max_prob > 50:
                confidence = "Medium"
            else:
                confidence = "Low"

            # --- Output ---
            print("=" * 40)
            print(f"Predicted Gesture: {prediction}")
            print(f"Confidence: {confidence} ({max_prob:.1f}%)")

            print("Probability distribution:")
            for gesture, prob in zip(model.classes_, probabilities):
                print(f"  Gesture {gesture}: {prob*100:.1f}%")

    except KeyboardInterrupt:
        print("\nStopping live prediction...")
        break
    except Exception as e:
        print("Error:", e)
