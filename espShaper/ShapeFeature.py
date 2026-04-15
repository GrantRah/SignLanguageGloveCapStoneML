"this takes the csv and then extracts features for each gesture block (defined by gestureID) making a block of data into a single row of features "
"for all of the columns in that one gesture id, it will then out a cvs of the features."
"then you gotta go in the cvs and assign the shape label to each gestureid manually in the cvs"
"this will then be used to traint the shape the model"
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
INPUT_CSV = "shape_data.csv"
OUTPUT_CSV = "shape_features.csv"

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()

print("Loaded data:", df.shape)


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


# =========================
# IMPROVED GRAVITY ESTIMATION (CRITICAL FOR TABLE DRAWING)
# =========================
def estimate_gravity(ax, ay, az, alpha=0.98):
    gx = np.zeros_like(ax)
    gy = np.zeros_like(ay)
    gz = np.zeros_like(az)

    gx[0], gy[0], gz[0] = ax[0], ay[0], az[0]

    for i in range(1, len(ax)):
        gx[i] = alpha * gx[i-1] + (1 - alpha) * ax[i]
        gy[i] = alpha * gy[i-1] + (1 - alpha) * ay[i]
        gz[i] = alpha * gz[i-1] + (1 - alpha) * az[i]

    ax = ax - gx
    ay = ay - gy
    az = az - gz

    return ax, ay, az


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(block):
    ax = block["Ax"].values
    ay = block["Ay"].values
    az = block["Az"].values

    # ======================================================
    # GRAVITY REMOVAL (stable for DRAWING ON TABLE)
    # ======================================================
    ax, ay, az = estimate_gravity(ax, ay, az)

    features = {}

    # ======================================================
    # BASIC STATISTICS
    # ======================================================
    for data, name in zip([ax, ay, az], ["x", "y", "z"]):
        features[f"mean_{name}"] = np.mean(data)
        features[f"var_{name}"] = np.var(data)
        features[f"std_{name}"] = np.std(data)
        features[f"mean_abs_{name}"] = np.mean(np.abs(data))

    # ======================================================
    # RMS / ENERGY
    # ======================================================
    features["rms_x"] = np.sqrt(np.mean(ax**2))
    features["rms_y"] = np.sqrt(np.mean(ay**2))
    features["rms_z"] = np.sqrt(np.mean(az**2))

    features["energy_x"] = np.sum(ax**2)
    features["energy_y"] = np.sum(ay**2)
    features["energy_z"] = np.sum(az**2)

    # ======================================================
    # PEAK FEATURES
    # ======================================================
    for data, name in zip([ax, ay, az], ["x", "y", "z"]):
        features[f"max_{name}"] = np.max(data)
        features[f"min_{name}"] = np.min(data)
        features[f"range_{name}"] = np.max(data) - np.min(data)

    # ======================================================
    # ZERO CROSSING RATE
    # ======================================================
    features["zc_x"] = zero_crossings(ax)
    features["zc_y"] = zero_crossings(ay)
    features["zc_z"] = zero_crossings(az)

    # ======================================================
    # CORRELATION (shape signature = VERY IMPORTANT)
    # ======================================================
    features["corr_xy"] = np.corrcoef(ax, ay)[0, 1] if len(ax) > 1 else 0
    features["corr_xz"] = np.corrcoef(ax, az)[0, 1] if len(ax) > 1 else 0
    features["corr_yz"] = np.corrcoef(ay, az)[0, 1] if len(ax) > 1 else 0

    # ======================================================
    # MAGNITUDE FEATURES (motion strength)
    # ======================================================
    mag = np.sqrt(ax**2 + ay**2 + az**2)

    features["mag_mean"] = np.mean(mag)
    features["mag_var"] = np.var(mag)
    features["mag_std"] = np.std(mag)
    features["mag_max"] = np.max(mag)
    features["mag_min"] = np.min(mag)
    features["mag_range"] = np.max(mag) - np.min(mag)

    # direction contribution ratios
    features["ax_ratio"] = np.mean(np.abs(ax)) / (np.mean(mag) + 1e-8)
    features["ay_ratio"] = np.mean(np.abs(ay)) / (np.mean(mag) + 1e-8)
    features["az_ratio"] = np.mean(np.abs(az)) / (np.mean(mag) + 1e-8)

    # ======================================================
    # LENGTH (time window size)
    # ======================================================
    features["length"] = len(ax)

    # ======================================================
    # JERK (motion sharpness)
    # ======================================================
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

    # ======================================================
    # PEAK TO PEAK
    # ======================================================
    features["ptp_x"] = np.max(ax) - np.min(ax)
    features["ptp_y"] = np.max(ay) - np.min(ay)
    features["ptp_z"] = np.max(az) - np.min(az)

    # ======================================================
    # SMOOTHNESS (how clean the drawing is)
    # ======================================================
    features["smooth_x"] = np.mean(np.abs(np.diff(ax)))
    features["smooth_y"] = np.mean(np.abs(np.diff(ay)))
    features["smooth_z"] = np.mean(np.abs(np.diff(az)))

    # ======================================================
    # TILT (NOW MEANINGFUL FOR TABLE DRAWING)
    # ======================================================
    # Instead of raw angles, we measure directional consistency
    features["tilt_xy"] = np.mean(np.arctan2(ay, ax))
    features["tilt_xz"] = np.mean(np.arctan2(az, ax))
    features["tilt_yz"] = np.mean(np.arctan2(az, ay))

    # ======================================================
    # ENTROPY (complexity of shape)
    # ======================================================
    features["entropy_x"] = signal_entropy(ax)
    features["entropy_y"] = signal_entropy(ay)
    features["entropy_z"] = signal_entropy(az)

    return features


# =========================
# BUILD DATASET
# =========================
feature_rows = []

group_col = "gesture_id" if "gesture_id" in df.columns else "Gesture"

for gid, block in df.groupby(group_col):
    feats = extract_features(block)

    feats["gesture_id"] = gid

    if "Gesture" in block.columns:
        feats["label"] = block["Gesture"].iloc[0]

    feature_rows.append(feats)

features_df = pd.DataFrame(feature_rows)

features_df.to_csv(OUTPUT_CSV, index=False)

print("\nDone. Saved to:", OUTPUT_CSV)
print(features_df.head())