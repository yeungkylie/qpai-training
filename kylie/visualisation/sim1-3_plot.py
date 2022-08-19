import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_gt_and_prediction_discrete(test_data, SET_NAME, process, num_oxy, array_shape):
    OUT_FOLDER = f"I:/research\seblab\data\group_folders\Kylie/validation/{test_data}/50000/"
    OUT_FILE = f"{SET_NAME}_{process}_validation.npz"
    data = np.load(os.path.join(OUT_FOLDER,OUT_FILE))
    ground_truth = data['ground_truth']
    prediction = data['prediction']
    mae = data['mae']
    oxy_vals = np.linspace(0,1,num_oxy)
    print(oxy_vals)
    pred_oxy=np.empty((num_oxy,array_shape))
    print(pred_oxy)
    for i in enumerate(oxy_vals):
        print(i)
        oxy = i[1]
        pred_oxy[i[0],:]=prediction[ground_truth==oxy]
        print(pred_oxy)
    return oxy_vals, pred_oxy, mae

sim = 3

num_oxy = 21

if sim == 1:
    test_data = "Simulation1_SingleVesselInWater"
    fig_title = "Simulation 1: Single Vessel In Water"
    best, best_process = "HighRes SmallVess", "thresholded_smoothed"
    best_title = "HighRes SmallVess (Thresholded and Smoothed)"
    worst, worst_process = "Heterogeneous with vessels", "thresholded_smoothed"
    worst_title = "Heterogeneous with vessels (Thresholded and Smoothed)"
    array_shape = 190400
elif sim == 2:
    test_data = "Simulation2_SingleVesselInBlood"
    fig_title = "Simulation 2: Single Vessel In Blood"
    best, best_process = "HighRes SmallVess", "thresholded_smoothed"
    best_title = "HighRes SmallVess (Thresholded and Smoothed)"
    worst, worst_process = "Heterogeneous with vessels", "thresholded_smoothed"
    worst_title = "Heterogeneous with vessels (Thresholded and Smoothed)"
    array_shape = 2094400
    num_oxy = 11
else:
    test_data = "Simulation3_VesselDeepInWater"
    fig_title = "Simulation 3: Vessel Deep in Water"
    best, best_process = "Acoustic", "smoothed"
    best_title = "Acoustic (Smoothed)"
    worst, worst_process = "Skin", "thresholded_smoothed"
    worst_title = "Skin (Thresholded and Smoothed)"
    array_shape = 20720

oxy_vals, pred1, mae1 = get_gt_and_prediction_discrete(test_data, best, best_process, num_oxy, array_shape)
oxy_vals, pred2, mae2 = get_gt_and_prediction_discrete(test_data, worst, worst_process, num_oxy, array_shape)
mpl.rcParams.update({'font.size': 9})
mpl.rcParams.update({'figure.titlesize': 13})
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)


fig, (ax, ax2) = plt.subplots(nrows=1,ncols=2, sharey=True)
fig.set_size_inches(10,4)
# best performing
parts = ax.violinplot(pred1.T, showmeans=False, showmedians=False, showextrema=False)
ax.set_title(best_title)

for pc in parts['bodies']:
    pc.set_facecolor('mediumseagreen')
    pc.set_edgecolor('mediumseagreen')
    pc.set_alpha(0.5)

quartile1, medians, quartile3 = np.percentile(pred1, [25, 50, 75], axis=1)

inds = np.arange(1, len(medians) + 1)
ax.vlines(inds, quartile1, quartile3, color='darkgreen', linestyle='-', lw=2)
ax.scatter(inds, medians, marker='o', color='k', s=30, zorder=1)
#plot the true value of oxygenation
ax.plot(oxy_vals * (num_oxy-1) + 1, oxy_vals, 'black', label="True value")
ax.set_xticks(np.linspace(1,num_oxy,6))
ax.set_xticklabels(np.linspace(0,100,6))
ax.set_yticks(np.linspace(0,1,6))
ax.set_yticklabels(np.linspace(0,100,6))
ax.set_xlabel("Ground truth oxygenation (%)")
ax.set_ylabel("Predicted oxygenation (%)")
ax.text(0.90, 0.1, f"{np.around(mae1*100, decimals=1)}%", transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)
ax.legend()

# worst performing
parts2 = ax2.violinplot(pred2.T, showmeans=False, showmedians=False, showextrema=False)
ax2.set_title(worst_title)

for pc in parts2['bodies']:
    pc.set_facecolor('lightcoral')
    pc.set_edgecolor('lightcoral')
    pc.set_alpha(0.5)

quartile1, medians, quartile3 = np.percentile(pred2, [25, 50, 75], axis=1)

inds = np.arange(1, len(medians) + 1)
ax2.vlines(inds, quartile1, quartile3, color='maroon', linestyle='-', lw=2)
ax2.scatter(inds, medians, marker='o', color='k', s=30, zorder=1)
#plot the true value of oxygenation
ax2.plot(oxy_vals * (num_oxy-1) + 1, oxy_vals, 'black', label="True value")
ax2.set_xticks(np.linspace(1,num_oxy,6))
ax2.set_xticklabels(np.linspace(0,100,6))
ax2.set_yticks(np.linspace(0,1,6))
ax2.set_yticklabels(np.linspace(0,100,6))
ax2.set_xlabel("Ground truth oxygenation (%)")
ax2.text(1.925, 0.1, f"{np.around(mae2*100, decimals=1)}%", transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

fig.suptitle(fig_title)
plt.tight_layout()
plt.savefig(f"I:/research\seblab\data\group_folders\Kylie\images\gt_vs_pred\{test_data}.png")
plt.show()