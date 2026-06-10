import yaml
import subprocess
from PyPDF2 import PdfMerger
import os
import numpy as np

fileNumber = 1
with open("file_to_skip.txt", "w") as f:
    f.write(str(fileNumber))

accuracies = []
f1_scores = []
auc_scores = []
ratios = []
while fileNumber <= 11: 
    ratio_result = subprocess.run(
        ["python3", "Classifier_our_data.py"],
        capture_output=True,
        text=True
    )
    result = subprocess.run(
        ["python3", "modell_test.py"],
        capture_output=True,
        text=True
    )

    lines = ratio_result.stdout.strip().splitlines()
    last_line = lines[-1]
    ratio = float(last_line)

    print("STDOUT:")
    print(result.stdout)

    print("STDERR:")
    print(result.stderr)

    print("RETURN CODE:")
    print(result.returncode)

    acc, f1, auc = map(float, result.stdout.strip().split(","))

    accuracies.append(acc)
    f1_scores.append(f1)
    auc_scores.append(auc)
    ratios.append(ratio)

    fileNumber = fileNumber +1
    if fileNumber ==5: fileNumber +=1
    print(f"fileNumber: {fileNumber}")
    with open("file_to_skip.txt", "w") as f:
        f.write(str(fileNumber))


print("Ratios: ", ratios)
print("Mean ratio: ", np.nanmean(ratios))
print("Std ratios: ", np.nanstd(ratios))

print("Accuracies:", accuracies)
print("F1 scores:", f1_scores)
print("AUC scores:", auc_scores)

print("Mean Accuracy:", np.nanmean(accuracies))
print("Std Accuracy :", np.nanstd(accuracies))

print("Mean F1:", np.nanmean(f1_scores))
print("Std F1 :", np.nanstd(f1_scores))

print("Mean AUC:", np.nanmean(auc_scores))
print("Std AUC :", np.nanstd(auc_scores))



with open("LOCV_results.txt", 'w') as f:
    f.write(f"""The accuracies are: {accuracies}
        Mean of the accuracies is: {np.mean(accuracies)}
        Standard deviation is: {np.std(accuracies)}
        """)

'''
pdf_folder = "./plots_LOCV/"
merger = PdfMerger()

for i in range(1, 20):  # 1 to 19
    file_path = os.path.join(pdf_folder, f"plot_{i}.pdf")
    if os.path.exists(file_path):
        merger.append(file_path)

merger.write("LOCV.pdf")
merger.close()
'''