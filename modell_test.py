import pandas as pd
import numpy as np
import joblib
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import skew, kurtosis
import yaml
from sklearn.metrics import accuracy_score

SPS = 250

with open("file_to_skip.txt", "r") as f:
    fileNumberToTest = f.read().strip()

def bandpass_filter(data, low_f, high_f, order=4):
    nyq = SPS / 2
    b, a = butter(order, [low_f/nyq, high_f/nyq], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, quality=30):
    freq = 50  # 50Hz Europe
    b, a = iirnotch(freq, quality, SPS)
    return filtfilt(b, a, data)

with open("window_size.yaml", "r") as f:
    config = yaml.safe_load(f)
    window_size = int(config["window_size"] * SPS)
    step_size = int(config["step_size"] * SPS)
    window_time = config["window_size"]

clf = joblib.load("blink_model.pkl")
scaler = joblib.load("scaler.pkl")

filler_zero = "0" if int(fileNumberToTest) < 10 else ""
dataFile = f"Our_data_classify/ADS1291_{filler_zero}{fileNumberToTest}.csv"
labelsFile = f"Our_data_classify/events_{filler_zero}{fileNumberToTest}.csv"

df = pd.read_csv(dataFile, comment="#", sep=",", skipinitialspace=True)
events = pd.read_csv(labelsFile, comment="#")
blink_times = np.array(events["event_elapsed_ms"] / 1000.0)

raw = df["ADS1291_EXG"].astype(float).values
raw = raw - np.mean(raw)
raw = bandpass_filter(raw, 0.5, 70)
raw = notch_filter(raw)

time = df["sample_index"].values / SPS

X_new = []
time_centers = []

for start in range(0, len(raw) - window_size, step_size):
    end = start + window_size
    window = raw[start:end]
    window_hamming = window * np.hamming(len(window))

    t_start = time[start]
    t_end = time[end - 1]
    time_centers.append((t_start + t_end) / 2)

    X = np.fft.rfft(window_hamming)
    freqs = np.fft.rfftfreq(len(window_hamming), 1/SPS)

    def band_power(low, high):
        mask = (freqs >= low) & (freqs <= high)
        return np.sum(np.abs(X[mask])**2)

    features = [
        np.mean(window),
        np.var(window),
        np.max(window) - np.min(window),
        skew(window),
        kurtosis(window),
        band_power(0.5, 4),
    ]
    X_new.append(features)

y_true = np.array([
    np.any(np.abs(blink_times - t) <= (window_time / 2))
    for t in time_centers
], dtype=int)

X_new = scaler.transform(X_new)
y_pred = clf.predict(X_new)

accuracy = accuracy_score(y_true, y_pred)
print(accuracy)