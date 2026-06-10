from stat import FILE_ATTRIBUTE_ENCRYPTED

import pywt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

SPS = 250  
#get signal data
#file_path = "EEG_offline_data_set/S01_data.csv"
#df = pd.read_csv(file_path, usecols=[0, 1], sep=';')

#file_path = "better_raw_EEG_data/session_Soheil_2_min.txt"  
file_path = "EEG_Offline_data_set/S01_data.csv"
#df = pd.read_csv(file_path, comment="#", sep=",", skipinitialspace=True)
df = pd.read_csv(file_path, usecols=[0, 1], sep=';')

raw = df["FP1"].astype(float).values
index = df["Time (s)"].values

'''
start_sec = 5
end_sec = 3
skippedSamples_start = int(start_sec * SPS)
skippedSamples_end = int(end_sec*SPS)
raw = raw[skippedSamples_start:-skippedSamples_end]
'''
raw = raw - np.mean(raw)

#perform dwt
cA, cD = pywt.dwt(raw, 'db4')
coeffs = pywt.wavedec(raw, 'db4', level=5)
labels = ['A5', 'D5']

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

for i in range(2):
    axes[i].plot(coeffs[i])
    axes[i].set_title(labels[i])
    axes[i].set_xlim(0, len(coeffs[i]))

plt.tight_layout()
plt.show()