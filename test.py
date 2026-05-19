import pandas as pd

file_path = "New_app_EEG_data/ADS1291_Fabian.csv"

df = pd.read_csv(file_path, comment="#", sep=",", skipinitialspace=True)
df = df.iloc[2750:].reset_index(drop=True)
df["sample_index"] = df.index

df.to_csv(file_path, index=False)