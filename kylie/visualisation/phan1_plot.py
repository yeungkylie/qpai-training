import numpy as np
import os
import matplotlib.pyplot as plt

def get_gt_and_prediction(test_data, SET_NAME, process):
    OUT_FOLDER = f"I:/research\seblab\data\group_folders\Kylie/validation/{test_data}/50000/"
    OUT_FILE = f"{SET_NAME}_{process}_validation.npz"
    data = np.load(os.path.join(OUT_FOLDER,OUT_FILE))
    ground_truth = data['ground_truth']
    print(ground_truth.shape)
    prediction = data['prediction']
    joined = np.vstack((ground_truth, prediction))
    print(joined.shape)
    print(joined)
    # joined.sort()
    # print(joined)
    return joined


test_data = "Phantom1_flow_phantom_no_melanin"
ground_truth1, prediction1 = get_gt_and_prediction(test_data, "Acoustic", "thresholded_smoothed")
ground_truth2, prediction2 = get_gt_and_prediction(test_data, "Heterogeneous 0-100", process=None)
ground_truth3, prediction3 = get_gt_and_prediction(test_data, "Point Illumination", "Smoothed")
# ground_truth3, prediction3 = get_gt_and_prediction(test_data, "Acoustic", "thresholded")

fig, ax = plt.subplots()
x = np.linspace(0,1,124722)
ax.plot(x, ground_truth1, linestyle='dashed', label="True value")
ax.plot(x, prediction1, color='g', label="Acoustic (Thresholded and Smoothed)", alpha=0.5)
# ax.plot(x, prediction2, color='darkred', label="Heterogeneous 0-100 (No processing)", alpha=0.5)
ax.plot(x, prediction3, color='r', label="Point Illunination (Smoothed)", alpha=0.5)
plt.xlabel("Oxygenation (%)")
plt.ylabel("Timesteps (%)")
ax.legend()
plt.title("Phantom 1: Flow Phantom (No Melanin)")
plt.savefig(f"I:/research\seblab\data\group_folders\Kylie\images\gt_vs_pred\{test_data}.png")
plt.show()