import matplotlib.pyplot as plt
from kylie.post_processing import prepare_simpa_simulations as p
import os

datasets = ["Baseline", "0.6mm Res", "1.2mm Res"]

fig, axs = plt.subplots(nrows=1, ncols=3)
fig.set_size_inches(10,4)
for SET in enumerate(datasets):
    SET_NAME = SET[1]
    print(f"Generating PCA plot for {SET_NAME}...")
    SPECTRA = f"I:/research\seblab\data\group_folders\Kylie\datasets/{SET_NAME}/{SET_NAME}_spectra.npz"
    r_wavelengths, r_oxygenations, r_spectra, \
    r_melanin_concentration, r_background_oxygenation, \
    r_distances, r_depths = p.load_spectra_file(SPECTRA)
    data = r_oxygenations*100
    plt.subplot(1, 3, int(SET[0]+1))
    plt.hist(data, bins=10, rwidth=0.75)
    plt.xlabel("Oxygenation [%]")
    plt.ylabel("Frequency")
    plt.title(f"{SET_NAME} dataset")
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
plt.tight_layout()
SAVE_PATH = f"I:/research\seblab\data\group_folders\Kylie\images\hist/oxy_hist.png"
plt.savefig(SAVE_PATH)
plt.show()