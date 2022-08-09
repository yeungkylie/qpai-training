from kylie.post_processing import prepare_simpa_simulations as p
import numpy as np

datasets = [
            # "Baseline",
            # "0.6mm Res", "1.2mm Res",
            # "5mm Illumination", "Point Illumination",
            # "BG 60-80", "BG 0-100",
            # "Heterogeneous with vessels",
            # "Heterogeneous 60-80",
            "Heterogeneous 0-100"]
            # "High Res", "HighRes SmallVess",
            # "Skin", "Acoustic"]

for SET_NAME in datasets:
    OUT_FILE = f"I:/research\seblab\data\group_folders\Kylie/datasets/{SET_NAME}/{SET_NAME}_spectra.npz"
    r_wavelengths, r_oxygenations, r_spectra, \
        r_melanin_concentration, r_background_oxygenation,\
        r_distances, r_depths, r_pca_components = p.load_spectra_file(OUT_FILE)
    print(np.mean(r_oxygenations))
    print(r_oxygenations.shape)
    print(r_spectra.shape)
    # p.visualise_spectra(r_spectra, r_oxygenations, r_melanin_concentration,
    #                   r_distances, r_depths, num_sO2_brackets=5, num_samples=300)