import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

SPS = 250  

file_path = "raw_EEG_data/session_1776887178749.txt"
df = pd.read_csv(file_path, sep='\t')

raw = df["EEG_CH1_RAW"].values
time = np.arange(len(raw)) / SPS

def bandpass_filter(data, low_f, high_f, fs, order=4):
    nyq = fs / 2
    low = low_f / nyq
    high = high_f / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

delta = bandpass_filter(raw, 0.5, 4, SPS)

window_sec = 5
window_samples = window_sec * SPS

num_batches = len(raw) // window_samples

for i in range(num_batches):
    start = i * window_samples
    end = start + window_samples

    plt.figure(figsize=(10, 4))
    #plt.plot(time[start:end], raw[start:end], label="Raw EEG", alpha=0.4)
    plt.plot(time[start:end], delta[start:end], label="Delta (0.5–4 Hz)", linewidth=2)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"EEG Segment {i+1} ({start/SPS:.1f}–{end/SPS:.1f}s)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"batches/plot_{i}")
