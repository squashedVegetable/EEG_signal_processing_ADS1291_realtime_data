import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, welch, spectrogram
from matplotlib.widgets import Button
import math
import sys


SPS = 250  
GAIN = 6 #can be changed in app/in Java
V_REF= 2.42 #default ref voltage
INT_NORMILISATION = (2**23) - 1

fileNumber = 1
fileNumber = int(sys.argv[1])

file_path = f"signal_files/Data_19_01/Messdaten_19_01/{fileNumber}.csv"
#file_path = f"Our_data_classify/Data_19_01/{fileNumber}.csv"
blinks_path = f"signal_files/Events_19_01/e{fileNumber}.csv"

#name = "Soheil"
#file_name = "ADS1291_" + name + ".csv"
#file_path = "New_app_EEG_data/" + file_name
#blinks_path = "New_app_EEG_data/events_" + name + ".csv"
df = pd.read_csv(file_path, comment="#", sep=",", skipinitialspace=True)
#blinks = pd.read_csv(blinks_path)
#blink_times = blinks["time_seconds"].values


#blinks["time_seconds"] = blinks["time_seconds"] - 0.2
#blink_times = blinks["time_seconds"].values

start_sec = 0
end_sec = 0
skippedSamples_start = int(start_sec * SPS)
skippedSamples_end = int(end_sec*SPS)

events = pd.read_csv(blinks_path, comment="#")
if "event_elapsed_ms" in events.columns:
    blink_times = list(events["event_elapsed_ms"] / 1000.0 - start_sec)
elif "time_seconds" in events.columns:
    blink_times = list(events["time_seconds"] - start_sec)

raw = df["ADS1291_EXG"].values
raw = df["ADS1291_EXG"].astype(float).values
index = df["sample_index"].values

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
filtered_signal = filtered_signal/(GAIN*INT_NORMILISATION)*V_REF *1e6

# ---- Plot ----
time_sec = np.arange(len(raw)) / SPS
fig, ax = plt.subplots(figsize=(15, 6))

def add_blink(event):
    global current_x

    blink_times.append(current_x)

    line = ax.axvline(current_x, color='red', linestyle='--', alpha=0.8, picker=5)
    line_to_index[line] = len(blink_times) - 1

    fig.canvas.draw_idle()

ax.plot(time_sec, filtered_signal)

ax.set_title(f"EEG - {fileNumber}")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (µV)")

ax_button = plt.axes([0.8, 0.01, 0.1, 0.05])
btn_add = Button(ax_button, "Add blink")
btn_add.on_clicked(add_blink)

# store line objects
lines = []
line_to_index = {}

for i, t in enumerate(blink_times):
    line = ax.axvline(t, color="red", linestyle="--", picker=5)
    line_to_index[line] = i

selected_line = None

#plt.figure(figsize=(12, 6))

#blink_times["t_sec"] -= start_sec

#plt.plot(time, filtered_signal, label="EEG signal")
#plt.plot(time, bp_0_5_20, label="Bandpass (0.5–20 Hz)", alpha=0.7)
#plt.plot(time, delta, label="Delta (0.5–4 Hz)", linewidth=2)

#for t in blink_times:
#    plt.axvline(x=t, color="red", linestyle="--", alpha=0.6)


def on_pick(event):
    global selected_line
    selected_line = event.artist

current_x = 0
def on_motion(event):
    global selected_line, current_x

    if event.xdata is not None:
        current_x = event.xdata  # ← always update, moved outside the drag check

    if selected_line is None or event.xdata is None:
        return

    selected_line.set_xdata([event.xdata, event.xdata])

    idx = line_to_index.get(selected_line)
    if idx is not None:
        blink_times[idx] = event.xdata

    fig.canvas.draw_idle()

def on_release(event):
    global selected_line
    selected_line = None

def on_key(event):
    global current_x

    if event.key == 'b':
        blink_times.append(current_x)

        line = ax.axvline(current_x, color='red', linestyle='--', picker=5)
        line_to_index[line] = len(blink_times) - 1

        fig.canvas.draw()

# CONNECT EVENTS

fig.canvas.mpl_connect("pick_event", on_pick)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"EEG Signal - {fileNumber}")
plt.legend()
ax.grid(True)

plt.savefig("Plot our data")
plt.show()

blink_df = pd.DataFrame({
    "event_timestamp_ms": (np.array(blink_times) + start_sec) * 1000.0,
    "event_elapsed_ms": (np.array(blink_times) + start_sec) * 1000.0,
    "blink": 1
})

blink_df.to_csv(
    blinks_path,
    index=False
)

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

'''
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

plt.figure()
plt.pcolormesh(time_centers)# shading='gouraud')

plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Sliding Window FFT (Spectrogram)')
plt.colorbar(label='Amplitude')
plt.show()


freqs, psd = welch(filtered_signal, SPS, nperseg=1024)

plt.figure(figsize=(10, 5))
plt.semilogy(freqs, psd)

plt.title("Power Spectral Density (Welch)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim(0.5, 20)
plt.grid(True)

plt.show()

'''