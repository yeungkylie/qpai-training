import numpy as np
import os
import matplotlib.pyplot as plt

def get_gt_and_prediction_discrete(test_data, SET_NAME, process):
    OUT_FOLDER = f"I:/research\seblab\data\group_folders\Kylie/validation/{test_data}/50000/"
    OUT_FILE = f"{SET_NAME}_{process}_validation.npz"
    data = np.load(os.path.join(OUT_FOLDER,OUT_FILE))
    ground_truth = data['ground_truth']
    prediction = data['prediction']
    oxy_vals = np.linspace(0,1,21)
    print(oxy_vals)
    pred_oxy=np.empty((21,190400))
    print(pred_oxy)
    for i in enumerate(oxy_vals):
        print(i)
        oxy = i[1]
        pred_oxy[i[0],:]=prediction[ground_truth==oxy]
        print(pred_oxy)
    return oxy_vals, pred_oxy

test_data = "Simulation1_SingleVesselInWater"

oxy_vals, pred1 = get_gt_and_prediction_discrete(test_data, "HighRes SmallVess", "thresholded_smoothed")
oxy_vals, pred2 = get_gt_and_prediction_discrete(test_data, "Heterogeneous with vessels", "thresholded_smoothed")


fig, (ax, ax2) = plt.subplots(nrows=2,ncols=1, sharex=True)
# best performing
parts = ax.violinplot(pred1.T, showmeans=False, showmedians=False, showextrema=False)
ax.set_title("HighRes SmallVess (Thresholded and Smoothed)")

for pc in parts['bodies']:
    pc.set_facecolor('mediumseagreen')
    pc.set_edgecolor('mediumseagreen')
    pc.set_alpha(0.5)

quartile1, medians, quartile3 = np.percentile(pred1, [25, 50, 75], axis=1)

inds = np.arange(1, len(medians) + 1)
ax.vlines(inds, quartile1, quartile3, color='darkgreen', linestyle='-', lw=2)
ax.scatter(inds, medians, marker='o', color='k', s=30, zorder=1)
#plot the true value of oxygenation
ax.plot(oxy_vals * 20 + 1, oxy_vals, 'black', label="True value")
ax.set_xticks(np.linspace(1,21,6))
ax.set_xticklabels(np.linspace(0,100,6))
ax.set_yticks(np.linspace(0,1,6))
ax.set_yticklabels(np.linspace(0,100,6))
plt.xlabel("Ground truth oxygenation (%)")
plt.ylabel("Predicted oxygenation (%)")
ax.legend()

# worst performing
parts2 = ax2.violinplot(pred2.T, showmeans=False, showmedians=False, showextrema=False)
ax2.set_title("Heterogeneous with vessels (Thresholded and Smoothed)")

for pc in parts2['bodies']:
    pc.set_facecolor('lightcoral')
    pc.set_edgecolor('lightcoral')
    pc.set_alpha(0.5)

quartile1, medians, quartile3 = np.percentile(pred2, [25, 50, 75], axis=1)

inds = np.arange(1, len(medians) + 1)
ax2.vlines(inds, quartile1, quartile3, color='maroon', linestyle='-', lw=2)
ax2.scatter(inds, medians, marker='o', color='k', s=30, zorder=1)
#plot the true value of oxygenation
ax2.plot(oxy_vals * 20 + 1, oxy_vals, 'black', label="True value")
ax2.set_xticks(np.linspace(1,21,6))
ax2.set_xticklabels(np.linspace(0,100,6))
ax2.set_yticks(np.linspace(0,1,6))
ax2.set_yticklabels(np.linspace(0,100,6))
plt.xlabel("Ground truth oxygenation (%)")
plt.ylabel("Predicted oxygenation (%)")

# plt.title("Simulation 1: Single Vessel In Water")
plt.savefig(f"I:/research\seblab\data\group_folders\Kylie\images\gt_vs_pred\{test_data}.png")
plt.show()