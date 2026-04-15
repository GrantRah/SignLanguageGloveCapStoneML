"""this reads live accerlation data once the gesture id changes(button released) that indicates that the gesutre is done, then the featrues
are extracted and fed to the pretrained model for predictions showing confidence, with the random forest model"""
import serial
import time
import numpy as np
import pandas as pd
import joblib

# =========================
# CONFIG
# =========================
PORT = "COM5"
BAUD_RATE = 115200

model = joblib.load("Trained_Shapes_model.joblib")
feature_names = joblib.load("feature_Shapes.joblib")

# =========================
# HELPERS
# =========================
def zero_crossings(signal):
    return np.sum(np.diff(np.sign(signal)) != 0)

def signal_entropy(signal, bins=10):
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist + 1e-12
    return -np.sum(hist * np.log(hist))

def remove_gravity(signal, alpha=0.95):
    gravity = np.zeros_like(signal)
    gravity[0] = signal[0]
    for i in range(1, len(signal)):
        gravity[i] = alpha * gravity[i-1] + (1 - alpha) * signal[i]
    return signal - gravity


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features_from_window(data_block):
    block = pd.DataFrame(data_block, columns=["gesture_id","Ax","Ay","Az"])

    ax = remove_gravity(block["Ax"].values)
    ay = remove_gravity(block["Ay"].values)
    az = remove_gravity(block["Az"].values)

    features = {}

    # BASIC STATS
    for data, name in zip([ax, ay, az], ["x", "y", "z"]):
        features[f"mean_{name}"] = np.mean(data)
        features[f"var_{name}"] = np.var(data)
        features[f"std_{name}"] = np.std(data)
        features[f"mean_abs_{name}"] = np.mean(np.abs(data))

    # RMS
    features["rms_x"] = np.sqrt(np.mean(ax**2))
    features["rms_y"] = np.sqrt(np.mean(ay**2))
    features["rms_z"] = np.sqrt(np.mean(az**2))

    # ENERGY
    features["energy_x"] = np.sum(ax**2)
    features["energy_y"] = np.sum(ay**2)
    features["energy_z"] = np.sum(az**2)

    # PEAK / RANGE
    for data, name in zip([ax, ay, az], ["x", "y", "z"]):
        features[f"max_{name}"] = np.max(data)
        features[f"min_{name}"] = np.min(data)
        features[f"range_{name}"] = np.max(data) - np.min(data)

    # ZERO CROSSINGS
    features["zc_x"] = zero_crossings(ax)
    features["zc_y"] = zero_crossings(ay)
    features["zc_z"] = zero_crossings(az)

    # CORRELATION
    features["corr_xy"] = np.corrcoef(ax, ay)[0, 1] if len(ax) > 1 else 0
    features["corr_xz"] = np.corrcoef(ax, az)[0, 1] if len(ax) > 1 else 0
    features["corr_yz"] = np.corrcoef(ay, az)[0, 1] if len(ax) > 1 else 0

    # MAGNITUDE
    mag = np.sqrt(ax**2 + ay**2 + az**2)

    features["mag_mean"] = np.mean(mag)
    features["mag_var"] = np.var(mag)
    features["mag_std"] = np.std(mag)
    features["mag_max"] = np.max(mag)
    features["mag_min"] = np.min(mag)
    features["mag_range"] = np.max(mag) - np.min(mag)

    features["ax_ratio"] = np.mean(np.abs(ax)) / (np.mean(mag) + 1e-8)
    features["ay_ratio"] = np.mean(np.abs(ay)) / (np.mean(mag) + 1e-8)
    features["az_ratio"] = np.mean(np.abs(az)) / (np.mean(mag) + 1e-8)

    # LENGTH
    features["length"] = len(ax)

    # JERK
    jerk_x = np.diff(ax)
    jerk_y = np.diff(ay)
    jerk_z = np.diff(az)

    features["jerk_mean_x"] = np.mean(jerk_x)
    features["jerk_mean_y"] = np.mean(jerk_y)
    features["jerk_mean_z"] = np.mean(jerk_z)

    features["jerk_std_x"] = np.std(jerk_x)
    features["jerk_std_y"] = np.std(jerk_y)
    features["jerk_std_z"] = np.std(jerk_z)

    features["jerk_energy_x"] = np.sum(jerk_x**2)
    features["jerk_energy_y"] = np.sum(jerk_y**2)
    features["jerk_energy_z"] = np.sum(jerk_z**2)

    # PEAK TO PEAK
    features["ptp_x"] = np.max(ax) - np.min(ax)
    features["ptp_y"] = np.max(ay) - np.min(ay)
    features["ptp_z"] = np.max(az) - np.min(az)

    # SMOOTHNESS
    features["smooth_x"] = np.mean(np.abs(np.diff(ax)))
    features["smooth_y"] = np.mean(np.abs(np.diff(ay)))
    features["smooth_z"] = np.mean(np.abs(np.diff(az)))

    # TILT
    features["tilt_xy"] = np.mean(np.arctan2(ay, ax))
    features["tilt_xz"] = np.mean(np.arctan2(az, ax))
    features["tilt_yz"] = np.mean(np.arctan2(az, ay))

    # ENTROPY
    features["entropy_x"] = signal_entropy(ax)
    features["entropy_y"] = signal_entropy(ay)
    features["entropy_z"] = signal_entropy(az)

    return features


# =========================
# SERIAL SETUP
# =========================
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
time.sleep(2)

print("Connected. Waiting for gestures...")

current_id = None
buffer = []

# =========================
# LIVE LOOP
# =========================
while True:
    try:
        line = ser.readline().decode("utf-8").strip()
        if not line:
            continue

        try:
            gid, ax, ay, az = line.split(",")
            gid = int(gid)
            ax, ay, az = float(ax), float(ay), float(az)
        except:
            continue

        # first gesture
        if current_id is None:
            current_id = gid

        # same gesture → keep collecting
        if gid == current_id:
            buffer.append([gid, ax, ay, az])

        # gesture ended → process window
        else:
            if len(buffer) > 15:

                feats = extract_features_from_window(buffer)

                # SAFE FEATURE ORDERING
                X_input = np.array([[feats.get(name, 0) for name in feature_names]])

                # PREDICT
                probs = model.predict_proba(X_input)[0]
                pred = model.predict(X_input)[0]
                confidence = np.max(probs)

                print("\n========================")
                print(f"Prediction: {pred}")
                print(f"Confidence: {confidence:.3f}")
                print("========================\n")

            # reset for new gesture
            buffer = [[gid, ax, ay, az]]
            current_id = gid

    except KeyboardInterrupt:
        print("\nStopped.")
        break