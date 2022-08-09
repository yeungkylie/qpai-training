import simpa as sp
import os
import glob
import numpy as np
from PIL import Image
from simpa.utils import get_data_field_from_simpa_output
from simpa.io_handling import load_hdf5
from simpa import Tags
import matplotlib.pyplot as plt


datasets = [
            "Baseline",
            "0.6mm Res", "1.2mm Res",
            "5mm Illumination", "Point Illumination",
            "BG 60-80", "BG 0-100",
            # "Heterogeneous with vessels",
            # # "Heterogeneous 60-80",
            # "Heterogeneous 0-100",
            "High Res", "HighRes SmallVess",
            "Skin",
            "Acoustic"]

# SAVE_PATH = f"I:/research\seblab\data\group_folders\Kylie\images\datasets/oxy/{SET_NAME}_oxy.png"

# if not os.path.exists(SAVE_PATH):
#     sp.visualise_data(path_to_hdf5_file=file[0],
#                       wavelength=700,
#                       show_oxygenation=True,
#                       # show_reconstructed_data=True,
#                       log_scale=False,
#                       show_xz_only=True,
#                       save_path=SAVE_PATH)

def all_oxy_images():
    data_to_show = []
    data_item_name = []
    for SET_NAME in datasets:
        path = f"I:/research\seblab\data\group_folders\Kylie/all simulated data/{SET_NAME}/"
        file = glob.glob(os.path.join(path, "*.hdf5"))
        file.sort()
        print(file)
        path_to_hdf5_file = file[0]
        file = load_hdf5(path_to_hdf5_file)
        wavelength = 700
        oxygenation = get_data_field_from_simpa_output(file, Tags.DATA_FIELD_OXYGENATION, wavelength)
        data_to_show.append(oxygenation)
        data_item_name.append(f"{SET_NAME}")

    print(oxygenation.shape)
    cmaps = "magma"
    num_rows = 5
    plt.figure(figsize=(3*4, num_rows*3.5))
    logscales = True

    for i in range(len(data_to_show)):
        plt.subplot(num_rows, len(data_to_show), i + 1)
        plt.title(data_item_name[i])
        # pos = int(np.shape(data_to_show[i])[1] / 2) - 1
        # data = np.rot90(data_to_show[i][:, pos, :], -1)
        data = np.rot90(data_to_show[i][:, :], -1)
        plt.imshow(np.log10(data) if logscales else data, cmap=cmaps)
        plt.colorbar()

if __name__ == "__main__":
    all_oxy_images()