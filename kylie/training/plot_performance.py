import train_random_forest as trf
import numpy as np
import matplotlib.pyplot as plt
import os

datasets = ["Baseline", "0.6mm Res", "1.2mm Res", "5mm Illumination",
            "Point Illumination", "BG 0-100", "BG 60-80",
            "Heterogeneous with vessels", "High Res",
            "HighRes SmallVess", "Point Illumination", "Skin"]

processes = [None, "thresholded", "smoothed", "noised", "thresholded_smoothed"]
all_mae = np.empty((len(processes), len(datasets)))  #create matrix to store metrics
all_mae_s = np.empty((len(processes), len(datasets)))
n_spectra1 = 400000
n_spectra2 = 77000

for process in enumerate(processes):
    for dataset in enumerate(datasets):
        score, mae = trf.load_metrics(dataset[1], n_spectra=n_spectra1, process=process[1])
        all_mae[process[0], dataset[0]] = mae
        score_s, mae_s = trf.load_metrics(dataset[1], n_spectra=n_spectra2, process=process[1])
        all_mae_s[process[0], dataset[0]] = mae_s
    fig, ax = plt.subplots()
    width = 0.35
    x = np.arange(len(datasets))
    rects1 = ax.bar(x-width/2, all_mae[process[0],:], width, label=f"{n_spectra1} spectra")
    rects2 = ax.bar(x+width/2, all_mae_s[process[0],:], width, label=f"{n_spectra2} spectra")
    ax.set(ylim=(0, 0.14))
    ax.set_xticks(x,datasets)
    if process[1] is None:
        plt.title('No processing')
    else:
        plt.title(process[1])
    plt.xticks(rotation=60)
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    ax.legend()
    fig.tight_layout()
    OUT_FILE = 'I:/research\seblab\data\group_folders\Kylie\images/metrics/'
    FILENAME = f'{process[1]}_mae.png'
    plt.savefig(os.path.join(OUT_FILE,FILENAME))
    plt.show()

print(all_mae)

