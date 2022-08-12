from kylie.post_processing import prepare_simpa_simulations as p
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


def distance_threshold(spectra, oxy, distances, depths, spacing=0.3):
        print("Applying distance threshold ...")
        print(f"oxy before: {oxy.shape}")
        print(f"spectra before: {spectra.shape}")
        print(distances)
        if spacing == 0.3:
                threshold_pixel = 2  # threshold of 2 pixels is equivalent to 0.6mm
        elif spacing == 0.15:
                threshold_pixel = 4  # threshold of 4 pixels is eqiuivalent to 0.6mm
        else:
                threshold_pixel = 1  # extract all outermost pixels
        oxy = oxy[distances <= threshold_pixel]
        spectra = spectra[:, distances <= threshold_pixel]
        depths = depths[distances <= threshold_pixel]
        distances = distances[distances <= threshold_pixel]
        print(f"oxy after: {oxy.shape}")
        print(f"spectra after: {spectra.shape}")
        print("Applying distance threshold ... [DONE]")
        return spectra, oxy, distances, depths


def noise_initial_pressure(spectra):
        print("Applying noise ...")
        mean = 1
        std = 0.05
        spectra = spectra * np.random.normal(mean, std, size=np.shape(spectra))
        print("Applying noise ... [DONE]")
        return spectra


def smooth_spectra(spectra):
        print("Applying smoothing ...")
        sigma = 1
        spectra = gaussian_filter1d(spectra, sigma)
        print("Applying smoothing ... [DONE]")
        return spectra

def visualise_processed_spectra(spectra, mode):
        """mode can be = distance, noise, smooth"""
        for idx in range(4):  # inspect how the processing changes the first 4 spectra
                print(f"Spectra {idx}")
                spectra_idx = spectra[:, idx]
                plt.subplot()
                plt.plot(np.linspace(700, 900, 41), spectra_idx,
                         color='b', linewidth=2, alpha=0.1)
                if mode == "noise":
                        spectra_new = noise_initial_pressure(spectra_idx)
                elif mode == "smooth":
                        spectra_new = smooth_spectra(spectra_idx)
                plt.plot(np.linspace(700, 900, 41), spectra_new,
                         color='r', linewidth=2, alpha=0.1)
                plt.show()

def generate_processed_datasets(SET_NAME, spacing = 0.3):
        IN_FILE = f"I:/research\seblab\data\group_folders\Kylie/datasets/{SET_NAME}/{SET_NAME}_spectra.npz"
        r_wavelengths, r_oxygenations, r_spectra, \
        r_melanin_concentration, r_background_oxygenation, \
        r_distances, r_depths, r_pca_components = p.load_spectra_file(IN_FILE)
        old = np.load(IN_FILE, allow_pickle=True)

        # extract distances
        thresholded_spectra, new_oxy, new_distances, new_depths \
                = distance_threshold(r_spectra, r_oxygenations, r_distances, r_depths, spacing=spacing)
        THRESHOLDED_FILE = f"I:/research\seblab\data\group_folders\Kylie/datasets/{SET_NAME}/{SET_NAME}_thresholded_spectra.npz"
        np.savez(THRESHOLDED_FILE,
                 wavelengths=old["wavelengths"],
                 oxygenations=new_oxy,
                 spectra=thresholded_spectra,
                 melanin_concentration=old["melanin_concentration"],
                 background_oxygenation=old["background_oxygenation"],
                 depths=new_depths,
                 distances=new_distances,
                 pca_components=old["pca_components"])

        # noised
        # noised_spectra = noise_initial_pressure(r_spectra)
        # NOISED_FILE = f"I:/research\seblab\data\group_folders\Kylie/datasets/{SET_NAME}/{SET_NAME}_noised_spectra.npz"
        # np.savez(NOISED_FILE,
        #          wavelengths=old["wavelengths"],
        #          oxygenations=old["oxygenations"],
        #          spectra=noised_spectra,
        #          melanin_concentration=old["melanin_concentration"],
        #          background_oxygenation=old["background_oxygenation"],
        #          depths=old["depths"],
        #          distances=old["distances"],
        #          pca_components=old["pca_components"])

        # smoothed
        smoothed_spectra = smooth_spectra(r_spectra)
        # SMOOTHED_FILE = f"I:/research\seblab\data\group_folders\Kylie/datasets/{SET_NAME}/{SET_NAME}_smoothed_spectra.npz"
        # np.savez(SMOOTHED_FILE,
        #          wavelengths=old["wavelengths"],
        #          oxygenations=old["oxygenations"],
        #          spectra=smoothed_spectra,
        #          melanin_concentration=old["melanin_concentration"],
        #          background_oxygenation=old["background_oxygenation"],
        #          depths=old["depths"],
        #          distances=old["distances"],
        #          pca_components=old["pca_components"])

        # thresholded and smoothed
        thresholded_smoothed_spectra, new_oxy, new_distances, new_depths \
                = distance_threshold(smooth_spectra(r_spectra), r_oxygenations, r_distances, r_depths, spacing=spacing)
        THRESHOLDED_SMOOTHED_FILE = f"I:/research\seblab\data\group_folders\Kylie/datasets/{SET_NAME}/{SET_NAME}_thresholded_smoothed_spectra.npz"
        np.savez(THRESHOLDED_SMOOTHED_FILE,
                 wavelengths=old["wavelengths"],
                 oxygenations=new_oxy,
                 spectra=thresholded_smoothed_spectra,
                 melanin_concentration=old["melanin_concentration"],
                 background_oxygenation=old["background_oxygenation"],
                 depths=new_depths,
                 distances=new_distances,
                 pca_components=old["pca_components"])
def plot_processed_spectra(SET_NAME, process):
    if process is None:
        OUT_FILE = f"I:/research\seblab\data\group_folders\Kylie/datasets/{SET_NAME}/{SET_NAME}_spectra.npz"
    else:
        OUT_FILE = f"I:/research\seblab\data\group_folders\Kylie/datasets/{SET_NAME}/{SET_NAME}_{process}_spectra.npz"
    r_wavelengths, r_oxygenations, r_spectra, \
        r_melanin_concentration, r_background_oxygenation,\
        r_distances, r_depths, r_pca_components = p.load_spectra_file(OUT_FILE)
    p.visualise_spectra(r_spectra, r_oxygenations, r_melanin_concentration,
                      r_distances, r_depths, num_sO2_brackets=5, num_samples=300, save_name=f"spectra/{SET_NAME}_{process}")


if __name__ == "__main__":
    processes = ["thresholded","smoothed","noised","thresholded_smoothed"]
    datasets = [
                "Baseline",
                "0.6mm Res", "1.2mm Res",
                "5mm Illumination",
                "Point Illumination",
                "BG 60-80", "BG 0-100",
                "Heterogeneous with vessels",
                # "Heterogeneous 60-80",
                # "Heterogeneous 0-100",
                "High Res", "HighRes SmallVess",
                "Skin", "Acoustic", "SmallVess"]
    for SET_NAME in datasets:
        print(f"Generating spectra plot for {SET_NAME}...")
        for process in processes:
            print(f"Generating spectra plot for {process}...")
            plot_processed_spectra(SET_NAME, process)
