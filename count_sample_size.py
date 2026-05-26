import pandas as pd

filepath = "Our_data_classify/ADS1291_"
fileNumber = 1

null_space = ""
num_rows = 0
while fileNumber < 12:
    if fileNumber >= 10:
        null_space = ""
    else: 
        null_space = "0"
    if fileNumber == 5:
        fileNumber+=1
        continue
    df = pd.read_csv(filepath + null_space + str(fileNumber) +".csv", comment="#", sep=",", skipinitialspace=True)
    num_rows += len(df)
    fileNumber+=1

print(num_rows)