import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib as mpl

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


test_data = "Phantom2_flow_phantom_medium_melanin"
ground_truth1, prediction1 = get_gt_and_prediction(test_data, "Acoustic", "thresholded_smoothed")
ground_truth3, prediction3 = get_gt_and_prediction(test_data, "Heterogeneous 0-100", "noised")
# ground_truth3, prediction3 = get_gt_and_prediction(test_data, "High Res", "Smoothed")
# ground_truth3, prediction3 = get_gt_and_prediction(test_data, "Acoustic", "thresholded")
mpl.rcParams.update({'font.size': 12})
fig, ax = plt.subplots()
x = np.linspace(0,1,98264)
ax.plot(x, ground_truth1, linestyle='dashed', label="True value")
ax.plot(x, gaussian_filter1d(prediction1, 200), color='g', label="Acoustic (Thresholded and Smoothed)", alpha=1)
ax.plot(x, prediction1, color='g', alpha=0.3)
ax.plot(x, gaussian_filter1d(prediction3, 200), color='r', label="High Resolution (Smoothed)")
ax.plot(x, prediction3, color='r', alpha=0.3)
plt.xlabel("Oxygenation (%)")
plt.ylabel("Timesteps (%)")
ax.legend()
plt.title("Phantom 2: Flow Phantom (Medium Melanin)")
plt.savefig(f"I:/research\seblab\data\group_folders\Kylie\images\gt_vs_pred\{test_data}.png")
plt.show()