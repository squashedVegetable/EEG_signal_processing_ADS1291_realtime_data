from sympy import fps
import yaml
import subprocess
from PyPDF2 import PdfMerger
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd


fileNumber = 1
with open("file_to_skip.txt", "w") as f:
    f.write(str(fileNumber))

SPS = 250

accuracies = []
f1_scores = []
auc_scores = []
ratios = []
confusion_matrices = []
precisions = []
recalls = []
fps_per_minute = []

fpr_all = []
tpr_all = []
curve_labels = []

while fileNumber <= 11:
    if fileNumber == 5:
        fileNumber += 1
        with open("file_to_skip.txt", "w") as f:
            f.write(str(fileNumber))
        continue

    ratio_result = subprocess.run(
        ["python3", "Classifier_our_data.py"],
        capture_output=True, text=True
    )
    result = subprocess.run(
        ["python3", "modell_test.py"],
        capture_output=True, text=True
    )

    lines = ratio_result.stdout.strip().splitlines()
    ratio = float(lines[-1])

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("RETURN CODE:", result.returncode)

    parts = result.stdout.strip().split(",")
    acc, f1, auc_var, tp, fp, fn, tn, precision, recall, fp_per_minute = map(float, [p.strip() for p in parts])
    confusion_matrices.append((int(tp), int(fp), int(fn), int(tn)))
    accuracies.append(acc)
    f1_scores.append(f1)
    auc_scores.append(auc_var)
    ratios.append(ratio)
    precisions.append(precision)
    recalls.append(recall)
    fps_per_minute.append(fp_per_minute)

    print(f"fileNumber: {fileNumber}")
    fileNumber += 1
    with open("file_to_skip.txt", "w") as f:
        f.write(str(fileNumber))

print("Ratios: ", ratios)
print("Mean ratio: ", np.nanmean(ratios))
print("Std ratios: ", np.nanstd(ratios))

print("Accuracies:", accuracies)
print("F1 scores:", f1_scores)
print("AUC scores:", auc_scores)
print("Confusion Matrices:", [f"{{{', '.join(map(str, cm))}}}" for cm in confusion_matrices])
print("Precisions:", [f"{p:.2f}" for p in precisions])
print("Recalls:", [f"{r:.2f}" for r in recalls])

print("Mean Accuracy:", np.nanmean(accuracies))
print("Std Accuracy :", np.nanstd(accuracies))
print("Mean Precision:", np.nanmean(precisions))
print("Std Precision :", np.nanstd(precisions))
print("Mean Recall:", np.nanmean(recalls))
print("Std Recall :", np.nanstd(recalls))

print("Mean F1:", np.nanmean(f1_scores))
print("Std F1 :", np.nanstd(f1_scores))

print("Mean AUC:", np.nanmean(auc_scores))
print("Std AUC :", np.nanstd(auc_scores))

print("FP per minute:", fps_per_minute)
print("Mean FP per minute:", np.nanmean(fps_per_minute))

tps = [cm[0] for cm in confusion_matrices]
fps = [cm[1] for cm in confusion_matrices]
fns = [cm[2] for cm in confusion_matrices]
tns = [cm[3] for cm in confusion_matrices]



with open("LOCV_results.txt", 'w') as f:
    f.write(f"""The accuracies are: {accuracies}
        Mean of the accuracies is: {np.mean(accuracies)}
        Standard deviation is: {np.std(accuracies)}
        The F1 scores are: {f1_scores}
        Mean of the F1 scores is: {np.mean(f1_scores)}
        Standard deviation is: {np.std(f1_scores)}
        The AUC scores are: {auc_scores}
        Mean of the AUC scores is: {np.mean(auc_scores)}
        Standard deviation is: {np.std(auc_scores)}
        The confusion matrices are: {confusion_matrices}
        The precisions are: {precisions}
        Mean of the precisions is: {np.mean(precisions)}
        Standard deviation is: {np.std(precisions)}
        The recalls are: {recalls}  
        Mean of the recalls is: {np.mean(recalls)}
        Standard deviation is: {np.std(recalls)}
        The FP per minute are: {fps_per_minute}
        Mean of the FP per minute is: {np.mean(fps_per_minute)}
        Standard deviation is: {np.std(fps_per_minute)}
        """)
    

plt.figure(figsize=(8, 6))

# Loop through 1 to 10 (matching your while loop logic)
for i in range(1, 11):
    if i == 5: continue # Skip file 5
    
    file_path = f"roc_data/run_{i}_predictions.csv"
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping.")
        continue
        
    try:
        df = pd.read_csv(file_path)
        y_true = df['y_true'].values
        y_prob = df['y_prob'].values
        
        # Calculate FPR and TPR for THIS specific run
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        
        # Store for mean calculation later
        fpr_all.append(fpr)
        tpr_all.append(tpr)
        
        # Plot individual curve (light gray)
        plt.plot(fpr, tpr, color='gray', lw=1, alpha=0.3, label=f'Fold {i}' if i==1 else "")
        
    except Exception as e:
        print(f"Error processing run {i}: {e}")

# Plot Random Chance Line
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

# --- Calculate and Plot Mean ROC Curve ---
# Interpolate all curves to the same x-axis (FPR) values
mean_fpr = np.unique(np.concatenate(fpr_all))
mean_tpr = []

for fpr_val in mean_fpr:
    # Interpolate TPR for this specific FPR across all runs
    tpr_interpolated = []
    for fpr_run, tpr_run in zip(fpr_all, tpr_all):
        tpr_interpolated.append(np.interp(fpr_val, fpr_run, tpr_run))
    mean_tpr.append(np.mean(tpr_interpolated))

mean_tpr = np.array(mean_tpr)
mean_auc = auc(mean_fpr, mean_tpr)

# Plot the thick average line
plt.plot(mean_fpr, mean_tpr, color='blue', lw=2, label=f'Mean ROC (AUC = {mean_auc:.2f})')

# Formatting
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curves for Blink Detection (10-Fold Cross-Validation)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# Save and Show
plt.savefig('LOCV_ROC_Curves_10_Folds.png', dpi=300)
plt.show()

print(f"Successfully plotted mean AUC: {mean_auc:.4f}")

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