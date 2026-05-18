import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, welch, spectrogram

SPS = 250  

file_path = "New_app_EEG_data/ADS1291.csv"
blinks_path = "New_app_EEG_data/events.csv"
df = pd.read_csv(file_path, comment="#", sep=",", skipinitialspace=True)
blinks = pd.read_csv(blinks_path, comment="#")

raw = df["ADS1291_EXG"].values
raw = df["ADS1291_EXG"].astype(float).values
index = df["sample_index"].values

start_sec = 0
end_sec = 0
skippedSamples_start = int(start_sec * SPS)
skippedSamples_end = int(end_sec*SPS)

if skippedSamples_end == 0:
    raw = raw[skippedSamples_start:]
else:
    raw = raw[skippedSamples_start:-skippedSamples_end]

time = np.arange(len(raw)) / SPS

#Filter 
def bandpass_filter(data, low_f, high_f, fs, order=4):
    nyq = fs / 2
    low = low_f / nyq
    high = high_f / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, quality=30):
    """Entfernt Netzstrom-Artefakte (50 Hz in Europa)"""
    freq = 50 #50Hz europe, 60Hz USA
    b, a = iirnotch(freq, quality, SPS)
    return filtfilt(b, a, data)

#remove DC offset. Optional i think maybe or something lol
raw = raw - np.mean(raw)
raw = notch_filter(raw)
filtered_signal = bandpass_filter(raw, 0.5, 5, SPS)

# ---- Plot ----
plt.figure(figsize=(12, 6))
t0_ms = blinks["event_timestamp_ms"].iloc[0]
t_start_ms = blinks["event_timestamp_ms"].iloc[0]   
blinks["t_sec"] = (blinks["event_timestamp_ms"] - t0_ms) / 1000.0
blinks["t_sec"] -= start_sec

plt.plot(time, filtered_signal, label="EEG signal")
#plt.plot(time, bp_0_5_20, label="Bandpass (0.5–20 Hz)", alpha=0.7)
#plt.plot(time, delta, label="Delta (0.5–4 Hz)", linewidth=2)

for _, row in blinks.iterrows():
    plt.axvline(x=row["t_sec"], color="red", linestyle="--", alpha=0.6)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("EEG Signal Comparison")
plt.legend()
plt.grid(True)

plt.savefig('plot_whole_signal')

plt.show()

'''
def band_power(low=0.5, high=20):
    mask = (fft_freqs >= low) & (fft_freqs <= high)
    return np.sum(np.abs(fft_results[mask])**2)
'''

#Frequency analysis

N = len(filtered_signal)

fft_vals = np.fft.rfft(filtered_signal)
fft_freqs = np.fft.rfftfreq(N, 1/SPS)

fft_magnitude = 2 * np.abs(fft_vals) / N

plt.figure(figsize=(10, 5))
plt.plot(fft_freqs, fft_magnitude)

plt.title("FFT of EEG Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.xlim(0.5, 20)

plt.show()

def band_power(low=0.5, high=20):
    mask = (fft_freqs >= low) & (fft_freqs <= high)
    return np.sum(np.abs(fft_vals[mask])**2)

freqs, times, Sxx = spectrogram(filtered_signal, SPS, nperseg=512, noverlap=256)

plt.figure(figsize=(12, 6))
plt.pcolormesh(times, freqs, Sxx, shading='auto')

plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram')
plt.colorbar(label='Power')

plt.ylim(0.5, 20)
plt.show()

#max_freq = 30 
#mask = freqs <= max_freq
#freqs_limited = freqs[mask]
#fft_limited = fft_results[mask, :]
'''
plt.figure()
plt.pcolormesh(time_centers)# shading='gouraud')

plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Sliding Window FFT (Spectrogram)')
plt.colorbar(label='Amplitude')
plt.show()
'''

freqs, psd = welch(filtered_signal, SPS, nperseg=1024)

plt.figure(figsize=(10, 5))
plt.semilogy(freqs, psd)

plt.title("Power Spectral Density (Welch)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim(0.5, 20)
plt.grid(True)

plt.show()