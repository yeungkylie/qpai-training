import numpy as np
import os
import matplotlib.pyplot as plt

def get_gt_and_prediction(test_data, SET_NAME, process):
    OUT_FOLDER = f"I:/research\seblab\data\group_folders\Kylie/validation/{test_data}/50000/"
    OUT_FILE = f"{SET_NAME}_{process}_validation.npz"
    data = np.load(os.path.join(OUT_FOLDER,OUT_FILE))
    ground_truth = data['ground_truth']
    prediction = data['prediction']
    joined = np.vstack((ground_truth, prediction))
    print(joined)
    joined.sort()
    print(joined)
    return joined


test_data = "Simulation5_ForearmInitialPressure"
ground_truth1, prediction1 = get_gt_and_prediction(test_data, "SmallVess", "thresholded_smoothed")
ground_truth2, prediction2 = get_gt_and_prediction(test_data, "Heterogeneous 60-80", process=None)
ground_truth3, prediction3 = get_gt_and_prediction(test_data, "Acoustic", "thresholded_smoothed")

fig, ax = plt.subplots()
ax.plot(ground_truth1, ground_truth1, linestyle='dashed', label="True value")
ax.plot(ground_truth1, prediction1, color='g', label="Small Vessels (Thresholded + Smoothed)")
ax.plot(ground_truth2, prediction2, color='darkred', label="Heterogeneous 60-80 (No processing)")
ax.plot(ground_truth3, prediction3, color='r', label="Acoustic (Thresholded + Smoothed)")
plt.xlabel("Ground truth oxygenation (%)")
plt.ylabel("Predicted oxygenation (%)")
ax.legend()
plt.title("Simulation 5: Forearm (Initial Pressure)")
plt.savefig(f"I:/research\seblab\data\group_folders\Kylie\images\gt_vs_pred\{test_data}.png")
plt.show()