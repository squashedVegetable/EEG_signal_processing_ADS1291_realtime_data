import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import sys
from scipy.signal import butter, filtfilt, iirnotch, stft
from scipy.stats import skew, kurtosis
import copy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import yaml

SPS = 250

#This is for LOCV
with open("file_to_skip.txt", "r") as f:
    fileNumberToTest = f.read().strip()

#Butterworth filter
def bandpass_filter(data, low_f, high_f, order = 4):
    nyq = SPS / 2      #Nyquisit frequency: Frequency until which can reliably be represented
    lowcut = low_f/nyq #0.5Hz highpass
    highcut = high_f/nyq #70Hz lowpass
    b, a = butter(order, [lowcut, highcut], btype='band')
    return filtfilt(b, a, data)

#optional, maybe not needed: COMMON MODE REJECTION filter or 50Hz filter for european signals
def notch_filter(data, quality=30):
    """Entfernt Netzstrom-Artefakte (50 Hz in Europa)"""
    freq = 50 #60Hz USA
    b, a = iirnotch(freq, quality, SPS)
    return filtfilt(b, a, data)

X_features = []
y_labels = []

#window, in which blinking is analized
#default settings
window_size = int(0.8 * SPS) 
step_size = int(0.1 * SPS)
with open("window_size.yaml", "r") as f:
    config = yaml.safe_load(f)
    window_size = int(config["window_size"] * SPS) #normally 0.8
    step_size = int(config["step_size"] * SPS) #normally 0.1

fileNumber = 1
while fileNumber <=11: 
    if str(fileNumber) == fileNumberToTest or fileNumber == 5:
        fileNumber = fileNumber +1
        continue
    if fileNumber < 10: 
        filler_zero = "0"
    else:
        filler_zero = ""
    dataFile = 'Our_data_classify/ADS1291_' + filler_zero + str(fileNumber) + '.csv'
    labelsFile = 'Our_data_classify/events_' + filler_zero + str(fileNumber) + '.csv'
    
    df = pd.read_csv(dataFile, comment="#", sep=",", skipinitialspace=True)
    events = pd.read_csv(labelsFile, comment="#")
    blink_times = list(events["event_elapsed_ms"] / 1000.0)

    raw = df["ADS1291_EXG"].astype(float).values
    raw = raw - np.mean(raw) #DC Offset
    raw = bandpass_filter(raw, 0.5, 70)
    raw = notch_filter(raw)

    time = df["sample_index"].values / SPS

    #brainwaves from Banpassfilter
    #delta = bandpass_filter(raw, 0.5, 4)
    #theta = bandpass_filter(df['FP1'].values, 4, 8)
    #alpha = bandpass_filter(df['FP1'].values, 8, 13)
    #beta = bandpass_filter(df['FP1'].values, 13, 30)
    #gamma = bandpass_filter(df['FP1'].values, 30, 100)

    #df.plot(x='Time (s)', y='FP1')

    #plt.plot(df['Time (s)'], delta, label='delta')
    #plt.plot(df['Time (s)'], theta, label='theta')
    #plt.plot(df['Time (s)'], alpha, label='alpha')
    #plt.plot(df['Time (s)'], beta, label='beta')
    #plt.plot(df['Time (s)'], gamma, label='gamma')

    def band_power(low, high):
        mask = (freqs >= low) & (freqs <= high)
        return np.sum(np.abs(X[mask])**2)
        #formula from a paper

    fft_results = []
    time_centers = []

    #loop for sliding window

    for start in range(0, len(raw) - window_size, step_size):
        end = start + window_size
        window = raw[start:end]
        window_hamming = window * np.hamming(len(window))

        t_start = time[start]
        t_end = time[end - 1]

        X = np.fft.rfft(window_hamming)
        freqs = np.fft.rfftfreq(len(window_hamming), 1 / SPS)

        def band_power(low, high):
            mask = (freqs >= low) & (freqs <= high)
            return np.sum(np.abs(X[mask]) ** 2)

        blink_times_arr = np.array(blink_times)

        label = int(np.any((blink_times_arr >= t_start) & (blink_times_arr <= t_end)))

        # skip ambiguous windows near a blink but not containing one
        if np.any((blink_times_arr >= t_start - 0.3) & (blink_times_arr <= t_end + 0.3)) and label == 0:
            continue

        y_labels.append(label)

        features = [
            np.mean(window),
            np.var(window),
            np.max(window) - np.min(window),
            skew(window),
            kurtosis(window),
            band_power(0.5, 4),
        ]
        X_features.append(features)

    '''
    for index, row in blinks.iterrows():
        if row['blink'] == 0:
            continue
        if any(start <= row['Time (s)'] <= end for start, end in corrupted):
            continue
        idx = (np.abs(df['Time (s)'] - row['Time (s)'])).argmin() #closest timestamp
        aligned_time = df['Time (s)'].iloc[idx]
        plt.axvline(x=aligned_time, linestyle = "--", color='red', alpha=0.5, label='Blink' if index == 0 else "")
        #idx = (np.abs(df['Time (s)'] - row['Time (s)'])).argmin()
    '''

    fft_results = np.array(fft_results).T  
    fileNumber = fileNumber+1


X_features = np.array(X_features)
y_labels = np.array(y_labels)

unique, counts = np.unique(y_labels, return_counts=True)

for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count}")

print("Percentage blink windows:", np.mean(y_labels) * 100)

n0 = np.sum(y_labels == 0)
n1 = np.sum(y_labels == 1)

print(f"Class 0 (no blink): {n0}")
print(f"Class 1 (blink): {n1}")
print(f"Ratio 0:1 = {n0/n1:.2f}")

scaler = StandardScaler()
X_features_scaled = scaler.fit_transform(X_features)

clf = RandomForestClassifier(n_estimators=400)
clf.fit(X_features_scaled, y_labels)

print("model trained")

joblib.dump(clf, "blink_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print(f"{n0/n1:.2f}")