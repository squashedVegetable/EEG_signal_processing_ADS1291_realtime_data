import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from matplotlib.backends.backend_pdf import PdfPages

SPS = 250

def bandpass_filter(data, low_f, high_f, order=4):
    nyq = SPS / 2
    b, a = butter(order, [low_f/nyq, high_f/nyq], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, quality=30):
    freq = 50  # 50Hz Europe
    b, a = iirnotch(freq, quality, SPS)
    return filtfilt(b, a, data)

fileNumber = 1
with PdfPages("all_plots.pdf") as pdf:
    while fileNumber <= 6:
        dataFile = f"Our_data_classify/ADS1291_0{fileNumber}.csv"
        labelsFile = f"Our_data_classify/events_0{fileNumber}.csv"

        df = pd.read_csv(dataFile, comment="#", sep=",", skipinitialspace=True)
        events = pd.read_csv(labelsFile, comment="#")
        blink_times = events["event_elapsed_ms"] / 1000.0

        raw = df["ADS1291_EXG"].astype(float).values
        raw = raw - np.mean(raw)
        raw = bandpass_filter(raw, 0.5, 70)
        raw = notch_filter(raw)

        time = df["sample_index"].values / SPS

        delta = bandpass_filter(raw, 0.5, 4)
        theta = bandpass_filter(raw, 4, 8)

        plt.plot(time, delta, label='delta')
        plt.plot(time, theta, label='theta')

        for i, t in enumerate(blink_times):
            plt.axvline(x=t, linestyle="--", color='red', alpha=0.5,
                        label='Blink' if i == 0 else "")

        plt.grid(True)
        plt.legend()
        plt.title(f"Plot {fileNumber}")
        pdf.savefig()
        plt.close()
        fileNumber += 1