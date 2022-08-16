import os
from kylie.post_processing import prepare_simpa_simulations as p
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from kylie.training import validation as v

def compute_spectra_plot(SET_NAME, save_name=None):
    print(f"Generating spectra plot for {SET_NAME}...")
    num_y_plots = 5
    num_sO2_brackets = 5
    num_samples = 300
    plt.figure(figsize=(4 * num_sO2_brackets, 4*num_y_plots))
    mpl.rcParams.update({'font.size': 12})
    processes = [None, "thresholded", "smoothed", "noised", "thresholded_smoothed"]
    process_name = ["No Processing",
                    "Thresholded",
                    "Smoothed",
                    "Noised",
                    "Thresholded and Smoothed"]
    for process in enumerate(processes):
        n = process[0]
        if process[1] is None:
            SPECTRA = f"I:/research\seblab\data\group_folders\Kylie\datasets/{SET_NAME}/{SET_NAME}_spectra.npz"
        else:
            SPECTRA = f"I:/research\seblab\data\group_folders\Kylie\datasets/{SET_NAME}/{SET_NAME}_{process[1]}_spectra.npz"
        r_wavelengths, oxy, r_spectra, \
        r_melanin_concentration, r_background_oxygenation, \
        distances, depths = p.load_spectra_file(SPECTRA)
        spectra = np.apply_along_axis(p.normalise_sum_to_one, 0, r_spectra)
        colouring = distances * depths

        for sO2_bracket in range(num_sO2_brackets):
            lower_so2_bound = sO2_bracket * (1 / num_sO2_brackets)
            upper_so2_bound = (sO2_bracket + 1) * (1 / num_sO2_brackets)
            selector = ((oxy >= lower_so2_bound) &
                        (oxy < upper_so2_bound))
            try:
                random_selection = np.random.choice(np.sum(selector), num_samples)
            except ValueError:
                random_selection = []
            so2_bracket_spectra = spectra[:, selector][:, random_selection]
            ax = plt.subplot(num_y_plots, num_sO2_brackets, num_sO2_brackets * n + sO2_bracket + 1)
            so2_bracket_colouring = colouring[selector][random_selection]
            if sO2_bracket == 0:
                plt.ylabel(process_name[n], {'size': 20})
            if n == 0:
                plt.title(f"{lower_so2_bound * 100:3.1f}% to {upper_so2_bound * 100:3.1f}% oxygenation",{'size': 14})
            for idx in range(len(so2_bracket_colouring)):
                plt.plot(np.linspace(700, 900, 41), so2_bracket_spectra[:, idx],
                         color=mpl.cm.magma(so2_bracket_colouring[idx]), linewidth=2, alpha=0.05)
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                if n == 4:
                    plt.xlabel("Wavelength [nm]")
            if sO2_bracket == num_sO2_brackets - 1:
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
                sm = plt.cm.ScalarMappable(cmap=mpl.cm.magma, norm=norm)
                sm.set_array([])
                cb = plt.colorbar(sm, ticks=np.linspace(0, 1, 11))
                cb.set_label("Spectral Colouring [a.u.]")
                cb.ax.set_yticks([1, 0.75, 0.5, 0.25, 0])
                cb.ax.set_yticklabels(['1', '0.25', '0.5', '0.25', '0'])
    # plt.title(f"{SET_NAME}")

    if save_name is not None:
        plt.savefig(save_name)
    plt.tight_layout()
    plt.show()


def compute_pca_plot(spectra, oxygenations, wavelengths, plot_title=None, save_path=None):

    PCA_PATH = r"I:\research\seblab\data\group_folders\Janek\learned_pa_oximetry\validation_data/pca/"

    if os.path.exists(PCA_PATH + f"baseline_pca_dump_{len(wavelengths)}.pt"):
        print("Loading fitted PCA...")
        pca = pickle.load(open(PCA_PATH + f"baseline_pca_dump_{len(wavelengths)}.pt", "rb"))
        data = np.load(PCA_PATH + f"baseline_spectra_{len(wavelengths)}.npz")
        baseline_spectra = data["baseline_spectra"]
        baseline_oxygenation = data["oxygenation"]
    else:
        BASELINE_FILE = f"I:/research\seblab\data\group_folders\Kylie\datasets/Baseline/Baseline_spectra.npz"
        r_wavelengths, r_oxygenations, r_spectra, \
        _, _, _, _, _, _, _, _, _ = p.load_spectra_file(BASELINE_FILE)
        wl_mask = [x in wavelengths for x in r_wavelengths]
        print("Normalising...")
        r_spectra = r_spectra[wl_mask, :]
        baseline_spectra = np.apply_along_axis(p.normalise_sum_to_one, 0, r_spectra)  # normalise
        print("Normalising...[Done]")
        print('Fitting pca...')
        pca = PCA(n_components=2)
        pca.fit(baseline_spectra.T)
        pickle.dump(pca, open(PCA_PATH + f"baseline_pca_dump_{len(wavelengths)}.pt", "wb"))
        print("Subsampling baseline spectra ...")
        spectra_choice = np.random.choice(np.shape(baseline_spectra)[1], 50000, replace=False)
        baseline_spectra = baseline_spectra[:, spectra_choice]
        baseline_oxygenation = r_oxygenations[spectra_choice]
        np.savez(PCA_PATH + f"baseline_spectra_{len(wavelengths)}.npz",
                 baseline_spectra=baseline_spectra,
                 oxygenation=baseline_oxygenation)
    print("Subsampling data spectra ...")
    spectra_choice = np.random.choice(np.shape(spectra)[1], 50000, replace=False)
    spectra = spectra[:, spectra_choice]
    oxygenations = oxygenations[spectra_choice]

    baseline_pca_components = pca.transform(baseline_spectra.T)
    data_pca_components = pca.transform(spectra.T)
    fig, ax = plt.subplots()

    light_cividis = cmap_map(lambda x: x / 2 + 0.5, mpl.cm.viridis)
    print(f"Shuffling baseline spectra ...")
    random_order = np.random.choice(len(baseline_pca_components), len(baseline_pca_components), replace=False)
    sc1 = ax.scatter(baseline_pca_components[random_order, 0], baseline_pca_components[random_order, 1],
                     c=baseline_oxygenation[random_order] * 100, cmap=light_cividis, s=10, alpha=0.75)
    cbar1 = plt.colorbar(sc1)
    cbar1.set_label("sO$_2$ of baseline spectra [%]")
    print(f"Shuffling data spectra ...")
    random_order = np.random.choice(len(data_pca_components), len(data_pca_components), replace=False)
    sc2 = ax.scatter(data_pca_components[random_order, 0], data_pca_components[random_order, 1],
                     c=oxygenations[random_order] * 100, cmap='magma', s=2, alpha=1)
    cbar2 = plt.colorbar(sc2)
    cbar2.set_label("sO$_2$ of target spectra [%]")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    if plot_title is not None:
        plt.title(plot_title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    old_LUT = np.asarray(cmap.colors)
    new_LUT = map(function, old_LUT)
    return mpl.colors.ListedColormap(list(new_LUT))


if __name__ == "__main__":
    datasets = [
                "Baseline",
                "0.6mm Res",
                "1.2mm Res",
                "5mm Illumination",
                "BG 0-100", "BG 60-80", "Point Illumination",
                "Heterogeneous with vessels",
                # "Heterogeneous 60-80",
                # "Heterogeneous 0-100",
                "High Res", "HighRes SmallVess", "Point Illumination", "Skin",
                "Acoustic", "SmallVess"]
    wavelengths = [700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900]
    # wavelengths = np.linspace(700,900,41)
    # SPECTRA = f"I:/research\seblab\data\group_folders\Kylie\datasets/Baseline/Baseline_spectra.npz"
    # r_wavelengths, r_oxygenations, r_spectra, \
    # r_melanin_concentration, r_background_oxygenation, \
    # r_distances, r_depths, r_pca_components = p.load_spectra_file(SPECTRA)
    in_silico = [
                "Simulation1_SingleVesselInWater",
                 "Simulation2_SingleVesselInBlood",
                 "Simulation3_VesselDeepInWater",
                 "Simulation4_HeterogeneousDistribution",
                 "Simulation5_ForearmInitialPressure",
                 "Simulation6_ForearmReconstructedData"]
    in_vitro = ["Phantom1_flow_phantom_no_melanin",
                "Phantom2_flow_phantom_medium_melanin"]
    for SET_NAME in datasets:
        SAVE_PATH = f"I:/research\seblab\data\group_folders\Kylie\images\processed_spectra/{SET_NAME}_processed.png"
        compute_spectra_plot(SET_NAME, save_name=SAVE_PATH)
    #     # SAVE_PATH = f"I:/research\seblab\data\group_folders\Kylie\images\pca/test_sets/{SET_NAME[1]}_pca_{len(wavelengths)}.png"
    #     if not os.path.exists(SAVE_PATH):
    #         print(f"Generating PCA plot for {SET_NAME}...")
    #         SPECTRA = f"I:/research\seblab\data\group_folders\Kylie\datasets/{SET_NAME}/{SET_NAME}_spectra.npz"
    #         # SPECTRA = f"I:/research\seblab\data\group_folders\Janek\learned_pa_oximetry/validation_data\in_vitro/{SET_NAME[1]}/{SET_NAME[1]}.npz"
    #         r_wavelengths, r_oxygenations, r_spectra, \
    #         r_melanin_concentration, r_background_oxygenation, \
    #         r_distances, r_depths, r_pca_components = p.load_spectra_file(SPECTRA)
    #         # r_spectra = v.filter_wavelengths(r_spectra)
    #         spectra = np.apply_along_axis(p.normalise_sum_to_one, 0, r_spectra)
    #         # compute_pca_plot(spectra, r_oxygenations, wavelengths, plot_title=f"{SET_NAME}", save_path=SAVE_PATH)
    #         compute_spectra_plot(spectra, r_oxygenations, r_melanin_concentration, r_distances, r_depths, num_samples=300, normalise=False)
    #     else:
    #         print(f"{SET_NAME}_pca.png already exists.")

### Kylie
    # baseline_spectra = np.apply_along_axis(p.normalise_sum_to_one, 0, v.filter_wavelengths(r_spectra))  # normalise
    # spectra_choice = np.random.choice(np.shape(baseline_spectra)[1], 50000, replace=False)
    # baseline_spectra = baseline_spectra[:,spectra_choice]
    # r_oxygenations = r_oxygenations[spectra_choice]
    # pca = PCA(n_components=2)
    # print(f'fitting pca')
    # pca.fit(baseline_spectra.T)
    # pca_components = pca.transform(baseline_spectra.T)
    # print('visualising baseline pca')
    # for SET_NAME in datasets:
    #     # first plot the baseline pca in light cividis
    #     TRANSFORMED_PCA = f'I:/research\seblab\data\group_folders\Kylie/transformed PCAs/{SET_NAME}_transformed_PCA.npz'
    #     if not os.path.exists(TRANSFORMED_PCA):
    #         random_order = np.random.choice(len(pca_components), len(pca_components), replace=False)
    #         fig, ax = plt.subplots()
    #         light_cividis = cmap_map(lambda x: x / 2 + 0.5, matplotlib.cm.viridis)
    #         sc_baseline = ax.scatter(pca_components[random_order, 0], pca_components[random_order, 1],
    #                                  c=r_oxygenations[random_order],
    #                                  cmap=light_cividis,
    #                                  s=5)
    #         # then load the spectra from the dataset
    #         OUT_FILE = f"I:/research\seblab\data\group_folders\Kylie\datasets/{SET_NAME}/{SET_NAME}_spectra.npz"
    #         m_wavelengths, m_oxygenations, m_spectra, \
    #         m_melanin_concentration, m_background_oxygenation, \
    #         m_distances, m_depths, m_pca_components = p.load_spectra_file(OUT_FILE)
    #         m_spectra = np.apply_along_axis(p.normalise_sum_to_one, 0, v.filter_wavelengths(m_spectra))
    #         spectra_choice = np.random.choice(np.shape(m_spectra)[1], 50000, replace=False)
    #         m_spectra = m_spectra[:, spectra_choice]
    #         m_oxygenations = m_oxygenations[spectra_choice]
    #
    #         # transform the pca on to baseline pca space
    #         print('transforming pca')
    #         pca1_components = pca.transform(m_spectra.T)
    #         np.savez(TRANSFORMED_PCA, pca=pca1_components, oxy=m_oxygenations)
    #         print('visualising pca')
    #         cm = plt.cm.magma
    #         random_order = np.random.choice(len(pca1_components), len(pca1_components), replace=False)
    #         sc = ax.scatter(pca1_components[random_order, 0], pca1_components[random_order, 1], c=m_oxygenations[random_order], cmap=cm, s=5)
    #         cb1 = plt.colorbar(sc_baseline)
    #         cb2 = plt.colorbar(sc)
    #         cb1.set_label("Baseline Oxygenation [%]")
    #         cb2.set_label("Oxygenation [%]")
    #         # plt.savefig(f'I:/research\seblab\data\group_folders\Kylie/transformed PCAs/{SET_NAME}_transformed_PCA.png')
    #         plt.show()
    #     else:
    #         fig, ax = plt.subplots()
    #         light_cividis = cmap_map(lambda x: x / 2 + 0.5, matplotlib.cm.viridis)
    #         sc_baseline = ax.scatter(pca_components[:, 0], pca_components[:, 1], c=r_oxygenations, cmap=light_cividis,
    #                                  s=5)
    #         # load transformed pca
    #         transformed_pca = np.load(TRANSFORMED_PCA)
    #         pca1_components = transformed_pca['pca']
    #         m_oxygenations = transformed_pca['oxy']
    #         print('visualising pca')
    #         cm = plt.cm.magma
    #         sc = ax.scatter(pca1_components[:, 0], pca1_components[:, 1], c=m_oxygenations, cmap=cm, s=5)
    #         cb1 = plt.colorbar(sc_baseline)
    #         cb2 = plt.colorbar(sc)
    #         cb1.set_label("Baseline Oxygenation [%]")
    #         cb2.set_label("Oxygenation [%]")
    #         # plt.savefig(f'I:/research\seblab\data\group_folders\Kylie/transformed PCAs/{SET_NAME}_transformed_PCA.png')
    #         plt.show()

