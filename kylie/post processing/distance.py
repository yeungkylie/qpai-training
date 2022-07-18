import numpy as np
import glob
import simpa as sp
from scipy.ndimage import distance_transform_edt
import os

def get_distances(folder_or_filename, target_tissue_class: int = 3):
    files, tmp_folder = get_files_and_tmp_folder_d(folder_or_filename)
    print(tmp_folder)
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    for file in files:
        filename = os.path.basename(file)
        if os.path.exists(tmp_folder + "/" + filename + ".npz"):
            print("Skipping", filename, "...")
            continue
        print(filename)

        segmentation_classes = sp.load_data_field(file, sp.Tags.DATA_FIELD_SEGMENTATION)
        if target_tissue_class == 3:
            distances = distance_transform_edt(segmentation_classes == target_tissue_class)
        else:
            distances = np.ones_like(segmentation_classes)

        distances = distances[segmentation_classes == target_tissue_class]

        np.savez(tmp_folder + "/" + filename + ".npz",
                 distances=distances)

def get_files_and_tmp_folder_d(folder_or_filename):
    files = []
    if os.path.isfile(folder_or_filename):
        files.append(folder_or_filename)
        foldername = os.path.dirname(folder_or_filename)
    elif os.path.isdir(folder_or_filename):
        files = glob.glob(os.path.join(folder_or_filename, "*.hdf5"))
        foldername = folder_or_filename
    else:
        raise FileNotFoundError("Input folder_or_filename must be a file or folder!")
    tmp_folder = foldername + "/distances/"
    return files, tmp_folder

def update_distances(folder_or_filename, target_file):
    _, tmp_folder = get_files_and_tmp_folder_d(folder_or_filename)

    if not os.path.exists(tmp_folder):
        print("No tmp folder found to extract npz files from...")
        return

    npz_files = glob.glob(tmp_folder + "/*.npz")
    distances = None

    for npz_file in npz_files:
        data = np.load(npz_file)
        _distances = data["distances"]
        if distances is None:
            distances = _distances
        else:
            distances = np.hstack([distances, _distances])

    # old file with wrong distances
    old = np.load("D:\Kylie Simulations\datasets/Baseline/Baseline_old.npz", allow_pickle=True)

    np.savez(target_file,
             wavelengths= old["wavelengths"],
             oxygenations= old["oxygenations"],
             spectra=old["spectra"],
             melanin_concentration=old["melanin_concentration"],
             background_oxygenation=old["background_oxygenation"],
             depths=old["depths"],
             distances=distances,
             pca_components=old["pca_components"])


if __name__ == "__main__":
    SET_NAME = "Baseline"
    IN_PATH = f"D:/Kylie Simulations/{SET_NAME}/"
    OUT_FILE = f"D:/Kylie Simulations/datasets/{SET_NAME}/{SET_NAME}_spectra.npz"
    get_distances(IN_PATH, target_tissue_class=3)
    update_distances(IN_PATH, OUT_FILE)
    # read_hdf5_and_extract_spectra(IN_PATH, target_tissue_class=3)
    # combine_spectra_files(IN_PATH, OUT_FILE)
    # r_wavelengths, r_oxygenations, r_spectra, \
    #     r_melanin_concentration, r_background_oxygenation,\
    #     r_distances, r_depths, r_pca_components = load_spectra_file(OUT_FILE)
    # visualise_spectra(r_spectra, r_oxygenations, r_melanin_concentration,
    #                   r_distances, r_depths, num_sO2_brackets=4, num_samples=300)
    # visualise_PCA(r_pca_components, r_oxygenations)