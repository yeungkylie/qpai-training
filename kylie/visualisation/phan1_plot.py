import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from kylie.post_processing import prepare_simpa_simulations as p
from sklearn.metrics import median_absolute_error

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
    return ground_truth, prediction

test_data = "Phantom1_flow_phantom_medium_melanin"
IN_FILE = f"I:/research\seblab\data\group_folders\Janek\learned_pa_oximetry/validation_data\in_vitro/{test_data}/{test_data}.npz"
print(f"Loading test data: {test_data} ...")
data = np.load(IN_FILE)
timesteps = data['timesteps']
lu = data['lu']
print(data['timesteps'])
distinct_timesteps = np.unique(timesteps)

# test_data = "Phantom1_flow_phantom_no_melanin"
ground_truth1, prediction1 = get_gt_and_prediction(test_data, "5mm Illumination", "smoothed")
# ground_truth3, prediction3 = get_gt_and_prediction(test_data, "Heterogeneous 0-100", process=None)
# ground_truth3, prediction3 = get_gt_and_prediction(test_data, "Point Illumination", "Smoothed")
ground_truth3, prediction3 = get_gt_and_prediction(test_data, "1.2mm Res", "smoothed")

lu_mae = median_absolute_error(lu, ground_truth1)
print(f"Linear Unmixing MAE: {lu_mae}")

print(ground_truth1.shape)
gt_sO2_mean = np.asarray([np.mean(ground_truth1[timesteps == step_value]) for step_value in distinct_timesteps])
pred_sO2_mean = np.asarray([np.mean(prediction1[timesteps == step_value]) for step_value in distinct_timesteps])
print(gt_sO2_mean.shape)
print(pred_sO2_mean.shape)

def plot_1():
    fig, ax = plt.subplots()
    x = timesteps
    ax.plot(x, ground_truth1, linestyle='dashed', label="True value")
    ax.plot(x, gaussian_filter1d(prediction1, 200), color='g', label="5mm Illumination (Smoothed)", alpha=1)
    ax.plot(x, prediction1, color='g', alpha=0.3)
    ax.plot(x, gaussian_filter1d(prediction3,200), color='r', label="1.2mm Res (Smoothed)")
    ax.plot(x, prediction3, color='r', alpha=0.3)
    ax.plot(x, gaussian_filter1d(lu,200), color='yellow', label="Linear Unmixing")
    ax.plot(x, lu, color='yellow', alpha=0.3)
    plt.xlabel("Timesteps [s]")
    plt.ylabel("Oxygenation [%]")
    ax.legend()
    plt.title("Phantom 1: Flow Phantom (Medium Melanin)")
    plt.savefig(f"I:/research\seblab\data\group_folders\Kylie\images\gt_vs_pred\{test_data}.png")
    plt.show()


def plot_2():
    fig, ax = plt.subplots()
    x = np.unique(timesteps)
    ax.plot(x, gt_sO2_mean, linestyle='dashed', label="True value")
    ax.plot(x, pred_sO2_mean, color='g', label="Acoustic (Thresholded and Smoothed)", alpha=1)
    # ax.plot(x, prediction1, color='g', alpha=0.3)
    # # ax.plot(x, prediction2, color='darkred', label="Heterogeneous 0-100 (No processing)", alpha=0.5)
    # ax.plot(x, gaussian_filter1d(prediction3,200), color='r', label="Point Illunination (Smoothed)")
    # ax.plot(x, prediction3, color='r', alpha=0.3)
    plt.xlabel("Timesteps (%)")
    plt.ylabel("Oxygenation (%)")
    ax.legend()
    plt.title("Phantom 1: Flow Phantom (Medium Melanin)")
    # plt.savefig(f"I:/research\seblab\data\group_folders\Kylie\images\gt_vs_pred\{test_data}.png")
    plt.show()


plot_1()