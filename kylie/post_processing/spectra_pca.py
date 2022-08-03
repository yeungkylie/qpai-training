from kylie.post_processing import prepare_simpa_simulations as p
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

BASELINE_FILE = f"I:/research\seblab\data\group_folders\Kylie\datasets/Baseline/Baseline_spectra.npz"
r_wavelengths, r_oxygenations, r_spectra, \
    r_melanin_concentration, r_background_oxygenation,\
    r_distances, r_depths, r_pca_components = p.load_spectra_file(BASELINE_FILE)
baseline_spectra = np.apply_along_axis(p.normalise_sum_to_one, 0, r_spectra)  # normalise
pca = PCA(n_components=2)
print(f'fitting pca')
pca.fit(baseline_spectra.T)
pca_components = pca.transform(baseline_spectra.T)
print('visualising baseline pca')

datasets = [
            # "0.6mm Res", "1.2mm Res", "5mm Illumination",
            # "BG 0-100", "BG 60-80", "Point Illumination", "BG 0-100", "BG 60-80",
            # "Heterogeneous with vessels", "Heterogeneous 60-80", "Heterogeneous 0-100",
            # "High Res",
            # "HighRes SmallVess", "Point Illumination", "Skin"
            "Acoustic"]

for SET_NAME in datasets:
    # first plot the baseline pca in grayscale
    fig, ax = plt.subplots()
    ax.scatter(pca_components[:, 0], pca_components[:, 1], c=r_oxygenations, cmap='gray', s=5)

    # then load the spectra from the dataset
    OUT_FILE = f"I:/research\seblab\data\group_folders\Kylie\datasets/{SET_NAME}/{SET_NAME}_spectra.npz"
    m_wavelengths, m_oxygenations, m_spectra, \
        m_melanin_concentration, m_background_oxygenation,\
        m_distances, m_depths, m_pca_components = p.load_spectra_file(OUT_FILE)
    m_spectra = np.apply_along_axis(p.normalise_sum_to_one, 0, m_spectra)

    # transform the pca on to baseline pca space
    print('transforming pca')
    pca1_components = pca.transform(m_spectra.T)
    np.savez(f'I:/research\seblab\data\group_folders\Kylie/transformed PCAs/{SET_NAME}_transformed_PCA.npz', pca=pca1_components)
    print('visualising pca')
    cm = plt.cm.plasma
    sc = ax.scatter(pca1_components[:, 0], pca1_components[:, 1], c=m_oxygenations, cmap=cm, s=5)
    plt.colorbar(sc)
    plt.savefig(f'I:/research\seblab\data\group_folders\Kylie/transformed PCAs/{SET_NAME}_transformed_PCA.png')
    plt.show()