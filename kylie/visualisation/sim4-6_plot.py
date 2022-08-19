import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_gt_and_prediction(test_data, SET_NAME, process):
    OUT_FOLDER = f"I:/research\seblab\data\group_folders\Kylie/validation/{test_data}/50000/"
    OUT_FILE = f"{SET_NAME}_{process}_validation.npz"
    data = np.load(os.path.join(OUT_FOLDER,OUT_FILE))
    ground_truth = data['ground_truth']
    prediction = data['prediction']
    mae = data['mae']
    joined = np.vstack((ground_truth, prediction))
    print(joined)
    joined.sort()
    print(joined)
    ground_truth = joined[0,:]
    prediction = joined[1,:]
    return ground_truth, prediction, mae

sim=4

if sim == 4:
    test_data = "Simulation4_HeterogeneousDistribution"
    plt_title = "Simulation 4: Heterogeneous Distribution"
    set1, process1, label1 = "BG 0-100", None, "Background 0-100 (No Processing)"
    set2, process2, label2 = "Heterogeneous 60-80", "noised", "Heterogeneous 60-80 (Noised)"
    set3, process3, label3 = "Skin", "noised", "Skin (Noised)"
elif sim == 5:
    test_data = "Simulation5_ForearmInitialPressure"
    plt_title = "Simulation 5: Forearm (Initial Pressure)"
    set1, process1, label1 = "SmallVess", "thresholded_smoothed", "Background 0-100 (No Processing)"
    set2, process2, label2 = "Heterogeneous 60-80", None, "Heterogeneous 60-80 (No processing)"
    set3, process3, label3 = "Acoustic", "thresholded_smoothed", "Acoustic (Thresholded + Smoothed)"
else:
    test_data = "Simulation6_ForearmReconstructedData"
    plt_title = "Simulation 6: Forearm (Reconstructed Data)"
    set1, process1, label1 = "Skin", "thresholded", "Skin (Thresholded)"
    set2, process2, label2 = "Heterogeneous 0-100", "smoothed", "Heterogeneous 0-100 (Smoothed)"
    set3, process3, label3 = "Heterogeneous with vessels", "smoothed", "Heterogeneous with vessels (Smoothed)"


ground_truth1, prediction1, mae1 = get_gt_and_prediction(test_data, set1, process1)
ground_truth2, prediction2, mae2 = get_gt_and_prediction(test_data, set2, process2)
ground_truth3, prediction3, mae3 = get_gt_and_prediction(test_data, set3, process3)

fig, ax = plt.subplots()
ax.plot(ground_truth1, ground_truth1, linestyle='dashed', label="True value")
ax.plot(ground_truth1, prediction1, color='g', label=f"{label1}: {np.around(mae1*100, decimals=1)}%")
ax.plot(ground_truth2, prediction2, color='darkred', label=f"{label2}: {np.around(mae2*100, decimals=1)}%")
ax.plot(ground_truth3, prediction3, color='r', label=f"{label3}: {np.around(mae3*100, decimals=1)}%")
plt.xlabel("Ground truth oxygenation (%)")
plt.ylabel("Predicted oxygenation (%)")
mpl.rcParams.update({'legend.fontsize':8})
ax.legend()
plt.title(plt_title)
plt.savefig(f"I:/research\seblab\data\group_folders\Kylie\images\gt_vs_pred\{test_data}.png")
plt.show()