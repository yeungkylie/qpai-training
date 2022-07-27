import os
import glob

import h5py
import simpa as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from kylie.training import train_random_forest as trf
from sklearn.decomposition import PCA


def normalise_sum_to_one(a):
    return a / np.linalg.norm(a)


def read_hdf5_and_extract_spectra(folder_or_filename, target_tissue_class: int = 3):
    files, tmp_folder = get_files_and_tmp_folder(folder_or_filename)
    print(tmp_folder)
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    files.sort()  #because for some reason it does not load in the right order
    for file in files:
        filename = os.path.basename(file)
        if os.path.exists(tmp_folder + "/" + filename + ".npz"):
            print("Skipping", filename, "...")
            continue
        print(filename)
        settings = sp.load_data_field(file, sp.Tags.SETTINGS)
        wavelengths = settings[sp.Tags.WAVELENGTHS]

        # segmentation_classes = sp.load_data_field(file, sp.Tags.DATA_FIELD_SEGMENTATION)
        data = h5py.File(file)
        segmentation_classes = np.array(data['new_segmentation'])
        if target_tissue_class == 3:
            distances = distance_transform_edt(segmentation_classes == target_tissue_class)
        else:
            distances = np.ones_like(segmentation_classes)

        depths = np.ones_like(distances)
        for z_layer in range(np.shape(depths)[2]):
            depths[:, :, z_layer] = z_layer
        depths = depths / np.max(depths)

        oxygenation = sp.load_data_field(file, sp.Tags.DATA_FIELD_OXYGENATION)

        initial_pressure = []
        for wavelength in wavelengths:
            initial_pressure.append(sp.load_data_field(file, sp.Tags.DATA_FIELD_INITIAL_PRESSURE, wavelength))
        initial_pressure = np.asarray(initial_pressure)

        # plt.figure()
        # plt.subplot(1, 4, 1)
        # plt.imshow(segmentation_classes[:, 20, :])
        # plt.subplot(1, 4, 2)
        # plt.imshow(distances[:, 20, :])
        # plt.subplot(1, 4, 3)
        # plt.imshow(depths[:, 20, :])
        # plt.subplot(1, 4, 4)
        # plt.imshow(oxygenation[:, 20, :])
        # plt.show()

        oxygenation_values = oxygenation[segmentation_classes == target_tissue_class]
        distances = distances[segmentation_classes == target_tissue_class]
        depths = depths[segmentation_classes == target_tissue_class]
        spectra = initial_pressure[:, segmentation_classes == target_tissue_class]

        np.savez(tmp_folder + "/" + filename + ".npz",
                 wavelengths=wavelengths,
                 oxygenation_values=oxygenation_values,
                 spectra=spectra,
                 depths=depths,
                 distances=distances)


def get_files_and_tmp_folder(folder_or_filename):
    files = []
    if os.path.isfile(folder_or_filename):
        files.append(folder_or_filename)
        foldername = os.path.dirname(folder_or_filename)
    elif os.path.isdir(folder_or_filename):
        files = glob.glob(os.path.join(folder_or_filename, "*.hdf5"))
        foldername = folder_or_filename
    else:
        raise FileNotFoundError("Input folder_or_filename must be a file or folder!")
    tmp_folder = foldername + "/deleteme/"
    return files, tmp_folder


def combine_spectra_files(folder_or_filename, target_file):
    if os.path.exists(target_file):
        print("Target file already exists... Skipping...")
        return

    _, tmp_folder = get_files_and_tmp_folder(folder_or_filename)

    if not os.path.exists(tmp_folder):
        print("No tmp folder found to extract npz files from...")
        return

    npz_files = glob.glob(tmp_folder + "/*.npz")
    wavelengths = None
    oxygenations = None
    spectra = None
    melanin_concentration = None
    background_oxygenation = None
    depths = None
    distances = None

    for npz_file in npz_files:
        data = np.load(npz_file)
        _oxygen = data["oxygenation_values"]
        _spectra = data["spectra"]
        _depths = data["depths"]
        _distances = data["distances"]
        if wavelengths is None:
            wavelengths = data["wavelengths"]
        if oxygenations is None:
            oxygenations = _oxygen
        else:
            oxygenations = np.hstack([oxygenations, _oxygen])

        if depths is None:
            depths = _depths
        else:
            depths = np.hstack([depths, _depths])

        if distances is None:
            distances = _distances
        else:
            distances = np.hstack([distances, _distances])

        if spectra is None:
            spectra = _spectra
        else:
            spectra = np.hstack([spectra, _spectra])

        if "_mel_" in npz_file:
            _melanin = float(npz_file.split("_mel_")[-1].split("_")[0].split(".hdf5")[0])
            _melanin = np.ones_like(_oxygen) * _melanin
            print("melanin:", _melanin)
            if melanin_concentration is None:
                melanin_concentration = _melanin
            else:
                melanin_concentration = np.hstack([melanin_concentration, _melanin])

        if "_bg_oxy_" in npz_file:
            _bg_oxy = float(npz_file.split("_bg_oxy_")[-1].split("_")[0].split(".hdf5")[0])
            _bg_oxy = np.ones_like(_oxygen) * _bg_oxy
            print("bg_oxy:", _bg_oxy)
            if background_oxygenation is None:
                background_oxygenation = _bg_oxy
            else:
                background_oxygenation = np.hstack([background_oxygenation, _bg_oxy])

    n_spectra = np.apply_along_axis(normalise_sum_to_one, 0,
                                    spectra)  # find principal components from normalised spectra
    pca = PCA(n_components=2)
    pca.fit(n_spectra.T)
    pca_components = pca.transform(n_spectra.T)
    # pca_components = 0

    folders = os.path.dirname(target_file)
    if not os.path.exists(folders):
        os.makedirs(folders)

    np.savez(target_file,
             wavelengths=wavelengths,
             oxygenations=oxygenations,
             spectra=spectra,
             melanin_concentration=melanin_concentration,
             background_oxygenation=background_oxygenation,
             depths=depths,
             distances=distances,
             pca_components=pca_components)


def load_spectra_file(file_path: str) -> tuple:
    print("Loading data...")
    data = np.load(file_path, allow_pickle=True)
    wavelengths = data["wavelengths"]
    oxygenations = data["oxygenations"]
    spectra = data["spectra"]
    distances = data["distances"]
    depths = data["depths"]
    melanin_concentration = data["melanin_concentration"]
    background_oxygenation = data["background_oxygenation"]
    pca_components = data["pca_components"]
    print("Loading data...[DONE]")
    return (wavelengths, oxygenations, spectra, melanin_concentration,
            background_oxygenation, distances, depths, pca_components)


def visualise_spectra(spectra, oxy, melanin, distances, depths, num_sO2_brackets=5, num_samples=100):
    print("Normalising data...")
    spectra = np.apply_along_axis(normalise_sum_to_one, 0, spectra)
    print("Normalising data...[Done]")
    num_y_plots = 2
    if melanin is not None and str(melanin) != "None":
        melanin = 1 - ((melanin - np.min(melanin)) / (np.max(melanin) - np.min(melanin)))
        num_y_plots = 3
    colouring = distances * depths
    plt.figure(figsize=(4 * num_sO2_brackets, 4*num_y_plots))
    for sO2_bracket in range(num_sO2_brackets):
        lower_so2_bound = sO2_bracket * (1 / num_sO2_brackets)
        upper_so2_bound = (sO2_bracket + 1) * (1 / num_sO2_brackets)
        selector = ((oxy >= lower_so2_bound) &
                    (oxy < upper_so2_bound))
        random_selection = np.random.choice(np.sum(selector), num_samples)
        so2_bracket_oxygenation_values = oxy[selector][random_selection]
        so2_bracket_spectra = spectra[:, selector][:, random_selection]

        plt.subplot(num_y_plots, num_sO2_brackets, sO2_bracket + 1)
        if sO2_bracket == 0:
            plt.ylabel("Coloured by sO2 value")
        plt.title(f"{lower_so2_bound * 100:3.1f}% to {upper_so2_bound * 100:3.1f}%")
        for idx in range(len(so2_bracket_oxygenation_values)):
            plt.plot(np.linspace(700, 900, 41), so2_bracket_spectra[:, idx],
                     color=mpl.cm.viridis(so2_bracket_oxygenation_values[idx]), linewidth=2, alpha=0.05)

        if sO2_bracket == num_sO2_brackets - 1:
            norm = mpl.colors.Normalize(vmin=0, vmax=100)
            sm = plt.cm.ScalarMappable(cmap=mpl.cm.viridis, norm=norm)
            sm.set_array([])
            cb = plt.colorbar(sm, ticks=np.linspace(0, 100, 11))
            cb.set_label("Blood oxygenation [%]")

        so2_bracket_colouring = colouring[selector][random_selection]
        plt.subplot(num_y_plots, num_sO2_brackets, num_sO2_brackets + sO2_bracket + 1)
        if sO2_bracket == 0:
            plt.ylabel("Coloured by Spectral Colouring")
        for idx in range(len(so2_bracket_colouring)):
            plt.plot(np.linspace(700, 900, 41), so2_bracket_spectra[:, idx],
                     color=mpl.cm.magma(so2_bracket_colouring[idx]), linewidth=2, alpha=0.05)

        if sO2_bracket == num_sO2_brackets - 1:
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap=mpl.cm.magma, norm=norm)
            sm.set_array([])
            cb = plt.colorbar(sm, ticks=np.linspace(0, 1, 11))
            cb.set_label("Spectral Colouring [a.u.]")

        if num_y_plots > 2:
            so2_bracket_melanin = melanin[selector][random_selection]
            plt.subplot(num_y_plots, num_sO2_brackets, 2 * num_sO2_brackets + sO2_bracket + 1)
            if sO2_bracket == 0:
                plt.ylabel("Coloured by (1-melanin volume fraction)")
            plt.xlabel("Wavelength [nm]")
            for idx in range(len(so2_bracket_melanin)):
                plt.plot(np.linspace(700, 900, 41), so2_bracket_spectra[:, idx],
                         color=mpl.cm.copper(so2_bracket_melanin[idx]), linewidth=2, alpha=0.05)

            if sO2_bracket == num_sO2_brackets - 1:
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
                sm = plt.cm.ScalarMappable(cmap=mpl.cm.copper, norm=norm)
                sm.set_array([])
                cb = plt.colorbar(sm, ticks=np.linspace(0, 1, 11))
                cb.set_label("(1-melanin concentration) [a.u.]")

    plt.tight_layout()
    plt.show()


def visualise_PCA(pca_components, colorcode):
    fig, ax = plt.subplots()
    cm = plt.cm.plasma
    sc = ax.scatter(pca_components[:, 0], pca_components[:, 1], c=colorcode, cmap=cm, s=5)
    plt.colorbar(sc)
    plt.show()

def extract_spectra(SET_NAME):
    print(f"--- extracting data from {SET_NAME} ---")
    IN_PATH = f"I:/research/seblab/data/group_folders/Kylie/all simulated data/{SET_NAME}/"
    OUT_FILE = f"I:/research/seblab/data/group_folders/Kylie/{SET_NAME}/{SET_NAME}_spectra.npz"
    # no target tissue class for vessel-less extraction
    read_hdf5_and_extract_spectra(IN_PATH)
    combine_spectra_files(IN_PATH, OUT_FILE)
    r_wavelengths, r_oxygenations, r_spectra, \
        r_melanin_concentration, r_background_oxygenation,\
        r_distances, r_depths, r_pca_components = load_spectra_file(OUT_FILE)
    visualise_spectra(r_spectra, r_oxygenations, r_melanin_concentration,
                      r_distances, r_depths, num_sO2_brackets=4, num_samples=300)
    visualise_PCA(r_pca_components, r_oxygenations)

if __name__ == "__main__":
    extract_spectra("Heterogeneous 60-80")
    trf.train_all("Heterogeneous 60-80", n_spectra=400000)
    trf.train_all("Heterogeneous 60-80", n_spectra=77000)
    extract_spectra("Heterogeneous 0-100")
    trf.train_all("Heterogeneous 0-100", n_spectra=400000)
    trf.train_all("Heterogeneous 0-100", n_spectra=77000)
