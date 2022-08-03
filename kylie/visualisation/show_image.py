import simpa as sp
import os
import glob

datasets = [
            # "Baseline",
            # "0.6mm Res", "1.2mm Res",
            # "5mm Illumination", "Point Illumination",
            # "BG 60-80", "BG 0-100",
            # "Heterogeneous with vessels", "Heterogeneous 60-80",
            # "Heterogeneous 0-100",
            # "High Res", "HighRes SmallVess",
            # "Skin",
            "Acoustic"]
for SET_NAME in datasets:
    path = f"I:/research\seblab\data\group_folders\Kylie/all simulated data/{SET_NAME}/"
    file = glob.glob(os.path.join(path, "*.hdf5"))
    file.sort()
    print(file)
    SAVE_PATH = f"I:/research\seblab\data\group_folders\Kylie\images\datasets/{SET_NAME}.png"

    if not os.path.exists(SAVE_PATH):
        sp.visualise_data(path_to_hdf5_file=file[0],
                          wavelength=700,
                          show_oxygenation=True,
                          show_initial_pressure=True,
                          show_absorption=True,
                          show_reconstructed_data=True,
                          log_scale=False,
                          show_xz_only=True,
                          save_path=SAVE_PATH)