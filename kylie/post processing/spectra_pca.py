from kylie import prepare_simpa_simulations as p

SET_NAME = "Point Illumination"
IN_PATH = f"D:/Kylie Simulations/{SET_NAME}/"
OUT_FILE = f"D:/Kylie Simulations/datasets/{SET_NAME}/{SET_NAME}_spectra.npz"
r_wavelengths, r_oxygenations, r_spectra, \
    r_melanin_concentration, r_background_oxygenation,\
    r_distances, r_depths, r_pca_components = p.load_spectra_file(OUT_FILE)
p.visualise_PCA(r_pca_components, r_oxygenations, flip_x=True)