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


test_data = "Simulation4_HeterogeneousDistribution"
ground_truth1, prediction1 = get_gt_and_prediction(test_data, "High Res", "smoothed")
ground_truth2, prediction2 = get_gt_and_prediction(test_data, "Heterogeneous 60-80", "noised")
ground_truth3, prediction3 = get_gt_and_prediction(test_data, "Skin", "noised")

fig, ax = plt.subplots()
ax.plot(ground_truth1, ground_truth1, linestyle='dashed', label="True value")
ax.plot(ground_truth1, prediction1, color='g', label="High Res (Smoothed)")
ax.plot(ground_truth2, prediction2, color='darkred', label="Heterogeneous 60-80 (Noised)")
ax.plot(ground_truth3, prediction3, color='r', label="Skin (Noised)")
plt.xlabel("Ground truth oxygenation (%)")
plt.ylabel("Predicted oxygenation (%)")
ax.legend()
plt.title("Simulation 4: Heterogeneous Distribution")
plt.savefig(f"I:/research\seblab\data\group_folders\Kylie\images\gt_vs_pred\{test_data}.png")
plt.show()