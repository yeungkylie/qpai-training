import simpa as sp
import os
import glob
import numpy as np
from PIL import Image
from simpa.utils import get_data_field_from_simpa_output
from simpa.io_handling import load_hdf5
from simpa import Tags
import matplotlib.pyplot as plt
import matplotlib as mpl


datasets = [
            # "Baseline"]
            # # "0.6mm Res",
            # # "1.2mm Res",
            # # "5mm Illumination", "Point Illumination",
            # # "BG 60-80", "BG 0-100",
            "Heterogeneous with vessels",
            # "Heterogeneous 60-80",
            "Heterogeneous 0-100"]
            # "High Res", "HighRes SmallVess",
            # "Skin",
            # "Acoustic"]


# if not os.path.exists(SAVE_PATH):
#     sp.visualise_data(path_to_hdf5_file=file[0],
#                       wavelength=700,
#                       show_oxygenation=True,
#                       # show_reconstructed_data=True,
#                       log_scale=False,
#                       show_xz_only=True,
#                       save_path=SAVE_PATH)

for SET_NAME in datasets:
    mpl.rcParams.update({'font.size': 12})
    path = f"I:/research\seblab\data\group_folders\Kylie/all simulated data/{SET_NAME}/"
    file = glob.glob(os.path.join(path, "*.hdf5"))
    file.sort()
    print(file)
    path_to_hdf5_file = file[0]
    file = load_hdf5(path_to_hdf5_file)
    wavelength = 700
    oxygenation = get_data_field_from_simpa_output(file, Tags.DATA_FIELD_OXYGENATION, wavelength)
    oxygenation = oxygenation * 100
    print(oxygenation.shape)
    cmaps = "magma"

    ticks = np.array([0, 10, 20, 30, 40, 50, 60])
    num_pix = 64
    sim_size = 19.2

    fig, ax = plt.subplots()
    plt.title(SET_NAME)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks * (sim_size / num_pix))
    ax.set_xlabel("X position [mm]")
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks * (sim_size / num_pix))
    ax.set_ylabel("Z position [mm]")
    pos = int(np.shape(oxygenation)[1] / 2) - 1
    data = np.rot90(oxygenation[:, pos, :], -1)
    plt.imshow(data, cmap=cmaps)
    cb = plt.colorbar()
    cb.set_label('Oxygenation [%]')
    SAVE_PATH = f"I:/research\seblab\data\group_folders\Kylie\images\datasets/oxy/{SET_NAME}_oxy.png"
    plt.savefig(SAVE_PATH)
    plt.tight_layout()
    plt.show()