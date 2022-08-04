import kylie.training.train_random_forest as trf
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_all_mae():
    for test_data in in_silico:
        ALL_METRICS = f"I:/research\seblab\data\group_folders\Kylie/validation\metrics/{test_data}_metrics.npz"
        if not os.path.exists(ALL_METRICS):
            for process in enumerate(processes):
                all_mae = np.empty((len(processes), len(datasets)))  # create matrix to store metrics
                n_spectra = 50000
                for dataset in enumerate(datasets):
                    try:
                        gt, prediction, ae, mae = trf.load_test_metrics(dataset[1], n_training_spectra=n_spectra, test_data=test_data, process=process[1])
                        all_mae[process[0], dataset[0]] = mae
                    except FileNotFoundError:
                        all_mae[process[0], dataset[0]] = 0  # some simulations do not have thresholded datasets, in which case set to 0
            np.savez(ALL_METRICS, all_mae=all_mae, processes=processes, datasets=datasets)
        else:
            metrics = np.load(ALL_METRICS)
            all_mae=metrics['all_mae']
        for process in enumerate(processes):
            print(f"Generating plot for {process}...")
            fig, ax = plt.subplots()
            width = 0.35
            x = np.arange(len(datasets))
            rects1 = ax.bar(x, all_mae[process[0],:], width)
            ax.set(ylim=(0, 0.25))
            ax.set_xticks(x,datasets)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if process[1] is None:
                plt.title(f'{test_data}; No processing')
            else:
                plt.title(f'{test_data}; {process[1]}')
            plt.xticks(rotation=60, ha='right')
            # ax.bar_label(rects1, padding=3)
            # ax.bar_label(rects2, padding=3)
            # ax.legend()
            fig.tight_layout()
            OUT_FILE = 'I:/research\seblab\data\group_folders\Kylie/validation/metrics/'
            FILENAME = f'{test_data}_{process[1]}_mae.png'
            plt.savefig(os.path.join(OUT_FILE,FILENAME))
            plt.show()


def plot_gt_vs_predicted(SET_NAME, process, n_spectra, test_data):
    gt, prediction, ae, mae = trf.load_test_metrics(SET_NAME, n_training_spectra=n_spectra, test_data=test_data,
                                                    process=process)

    fig, ax = plt.subplots()
    ax.scatter(gt, prediction, linewidths=0.05, alpha=0.002)
    plt.xlabel("Ground Truth Oxygenation")
    plt.ylabel("Predicted oxygenation")

    plt.show()


if __name__ == "__main__":
    datasets = ["Baseline",
                "0.6mm Res", "1.2mm Res",
                "5mm Illumination", "Point Illumination",
                "BG 60-80", "BG 0-100",
                "Heterogeneous with vessels", "Heterogeneous 60-80",
                "Heterogeneous 0-100",
                "High Res", "HighRes SmallVess",
                "Skin", "Acoustic"]

    processes = [None, "thresholded", "smoothed", "noised", "thresholded_smoothed"]

    in_silico = [
        # "Simulation1_SingleVesselInWater",
        "Simulation2_SingleVesselInBlood",
        "Simulation3_VesselDeepInWater",
        "Simulation4_HeterogeneousDistribution"]


    plot_all_mae()
    # plot_gt_vs_predicted("Baseline", None, 50000, "Simulation1_SingleVesselInWater")