from kylie import prepare_simpa_simulations as p
import simpa as sp
import scipy.ndimage.gaussianfilter1d

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