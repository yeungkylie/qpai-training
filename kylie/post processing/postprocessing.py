from kylie import prepare_simpa_simulations as p
import simpa as sp
import scipy.ndimage.gaussianfilter1d
import matplotlib.pyplot as plt

r_wavelengths, r_oxygenations, r_spectra, \
        r_melanin_concentration, r_background_oxygenation,\
        r_distances, r_depths, r_pca_components = p.load_spectra_file(OUT_FILE)

def distance_threshold(spectra, oxy, distances, spacing=0.3):
        threshold_pixel=1  # extract outermost pixels in any spacing
        threshold_mm=0.6   # and anything within 0.6mm
        print(distances*spacing)
        oxy = oxy[distances <= threshold_pixel | distances*spacing <= threshold_mm]
        spectra = spectra[distances <= threshold_pixel | distances*spacing <= threshold_mm]
        return spectra, oxy


def noise_initial_pressure(spectra):
        mean = 1
        std = 0.01
        spectra = spectra * np.random.normal(mean, std, size=np.shape(spectra))
        return spectra


def smooth_spectra(spectra):
        sigma = 1
        spectra = gaussianfilter1d(spectra,sigma)
        return spectra

def visualise_processed_spectra(spectra, mode, distance=None):
        """mode can be = distance, noise, smooth"""
        for idx in range(4):  # inspect how the processing changes the first 4 spectra
                spectra = [:,idx]
                plt.subplot()
                plt.plot(np.linspace(700, 900, 41), spectra,
                         color=blue, linewidth=2, alpha=0.05)
                if mode == distance:
                        spectra_new = distance_threshold(spectra, oxy, distances)
                elif mode == noise:
                        spectra_new = noise_initial_pressure(spectra)
                elif mode == smooth:
                        spectra_new = smooth_spectra(spectra)
                plt.plot(np.linspace(700, 900, 41), spectra_new,
                         color=red, linewidth=2, alpha=0.05)
                plt.show()