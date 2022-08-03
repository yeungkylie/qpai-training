from sklearn.ensemble import RandomForestRegressor
from kylie.post_processing import prepare_simpa_simulations as p
from sklearn.metrics import median_absolute_error
import numpy as np
import os
from sklearn.model_selection import KFold
import pickle
import time
start_time = time.time()

def get_normalised_spectra_oxy(SET_NAME, n_spectra, process, visualise, flowphantom):
    if process is None:
        IN_FILE = f"I:/research\seblab\data\group_folders\Kylie/datasets/{SET_NAME}/{SET_NAME}_spectra.npz"
    else:
        IN_FILE = f"I:/research\seblab\data\group_folders\Kylie/datasets/{SET_NAME}/{SET_NAME}_{process}_spectra.npz"
    print(f"Retreived training dataset {IN_FILE}")
    r_wavelengths, r_oxygenations, r_spectra, \
    r_melanin_concentration, r_background_oxygenation, \
    r_distances, r_depths, r_pca_components = p.load_spectra_file(IN_FILE)

    if flowphantom:
        flowphantom_wavelengths = np.asarray([700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900])
        wavelength_idx = (flowphantom_wavelengths - 700) / 5
        r_spectra = r_spectra[wavelength_idx.astype(int), :]
    print(f"Original spectra and oxy shapes: {r_spectra.shape}, {r_oxygenations.shape}.")

    # subsample to standardize num spectra in all datasets
    np.random.seed(1)  # use random seed for reproducibility
    num_samples = n_spectra
    random_selection = np.random.choice(np.size(r_spectra, axis=1), num_samples, replace=False)
    print(f"The following subset of spectra were selected: {random_selection}")
    r_oxygenations = r_oxygenations[random_selection]
    r_spectra = r_spectra[:, random_selection]
    print(f"New spectra and oxy shapes: {r_spectra.shape}, {r_oxygenations.shape}.")

    # normalise and transpose to fit random forest dimensions
    print(f"Normalising spectra ...")
    spectra = (np.apply_along_axis(p.normalise_sum_to_one, 0, r_spectra)).T
    print(f"spectra: {spectra}")
    oxy = r_oxygenations
    print(f"oxy: {oxy}")
    if visualise:
        print(f"Visualise training spectra...")
        p.visualise_spectra(spectra.T, oxy, r_melanin_concentration,
                          r_distances[random_selection], r_depths[random_selection], num_sO2_brackets=1, num_samples=300, normalise=False)
    return spectra, oxy

def train_random_forests(SET_NAME, n_spectra, flowphantom, process=None, visualise=False):
    """ process = 'thresholded', 'noised','smoothed' """
    print(f"Training {SET_NAME}:")
    spectra, oxy = get_normalised_spectra_oxy(SET_NAME, n_spectra, process, visualise, flowphantom)
    scores = []
    mae = []
    kf = KFold(n_splits=5, shuffle=False)
    for idx, (train_index, test_index) in enumerate(kf.split(spectra)):
        print("Train Index: ", train_index, "\n")
        print("Test Index: ", test_index)
        xtrain, xtest, ytrain, ytest = spectra[train_index], spectra[test_index], oxy[train_index], oxy[test_index]
        rfr = RandomForestRegressor(n_estimators=64, verbose=2, max_depth=16, n_jobs=-1)
        rfr.fit(xtrain, ytrain)

        scores.append(rfr.score(xtrain, ytrain))
        print(f"Fold {idx} scores: ", scores)  # shows how closely the target oxy can be regressed to spectra

        ypred = rfr.predict(xtest)
        mae.append(median_absolute_error(ytest, ypred))
        print(f"Fold {idx} median absolute error: ", mae)

        if not flowphantom:
            OUT_FOLDER = f"I:/research\seblab\data\group_folders\Kylie\Trained Models {n_spectra}/{SET_NAME}/{process}"
        else:
            OUT_FOLDER = f"I:/research\seblab\data\group_folders\Kylie\Trained Models {n_spectra} flowphantom/{SET_NAME}/{process}"
        if not os.path.exists(OUT_FOLDER):
            os.makedirs(OUT_FOLDER)
        OUT_FILE = f"{SET_NAME}_{process}_rf{idx}.sav"
        pickle.dump(rfr, open(os.path.join(OUT_FOLDER, OUT_FILE), 'wb'))  # save each fold
    mean_score = np.mean(scores)
    mean_mae = np.mean(mae)
    np.savez(os.path.join(OUT_FOLDER, f"{SET_NAME}_{process}_metrics.npz"),
             scores=scores, mae=mae, mean_score=mean_score, mean_mae=mean_mae)
    print(f"5-fold averaged score: {np.mean(scores)}")  # metric combining r-score of all folds
    print(f"5-fold averaged median absolute error: {np.mean(mae)}")  # metric combining mae of all folds


def train_all(SET_NAME, n_spectra, flowphantom=False):
    train_random_forests(SET_NAME, n_spectra, flowphantom=flowphantom)
    try:
        train_random_forests(SET_NAME, n_spectra, flowphantom, process="thresholded")
    except FileNotFoundError:
        print("Thresholded spectra does not exist")
        pass
    train_random_forests(SET_NAME, n_spectra, flowphantom, process="smoothed")
    train_random_forests(SET_NAME, n_spectra, flowphantom, process="noised")
    try:
        train_random_forests(SET_NAME, n_spectra, flowphantom, process="thresholded_smoothed")
    except FileNotFoundError:
        print("Thresholded spectra does not exist")
        pass

def load_metrics(SET_NAME, n_spectra, process=None):
    OUT_FOLDER = f"I:/research\seblab\data\group_folders\Kylie\Trained Models {n_spectra}/{SET_NAME}/{process}"
    metrics = np.load(os.path.join(OUT_FOLDER, f"{SET_NAME}_{process}_metrics.npz"))
    score = metrics["mean_score"]
    mae = metrics["mean_mae"]
    return score, mae

def load_all_metrics(SET_NAME, n_spectra):
    print(f"{SET_NAME} ({n_spectra} spectra-trained model) metrics:")
    score, mae = load_metrics(SET_NAME, n_spectra)
    # score_thresholded, mae_thresholded = load_metrics(SET_NAME, n_spectra, process="thresholded")
    score_smoothed, mae_smoothed = load_metrics(SET_NAME, n_spectra, process="smoothed")
    score_noised, mae_noised = load_metrics(SET_NAME, n_spectra, process="noised")
    # score_thresholded_smoothed, mae_thresholded_smoothed = load_metrics(SET_NAME, n_spectra,
    #                                                                             process="thresholded_smoothed")

    print(f"No processing: score = {score}, mae = {mae}")
    # print(f"Thresholded: score = {score_thresholded}, mae = {mae_thresholded}")
    print(f"Smoothed: score = {score_smoothed}, mae = {mae_smoothed}")
    print(f"Noised: score = {score_noised}, mae = {mae_noised}")
    # print(f"Thresholded and smoothed: score = {score_thresholded_smoothed}, mae = {mae_thresholded_smoothed}")


if __name__ == "__main__":
    datasets = ["Baseline", "0.6mm Res", "1.2mm Res", "5mm Illumination",
                "Point Illumination", "BG 0-100", "BG 60-80",
                "Heterogeneous with vessels", "Heterogeneous 60-80",
                "Heterogeneous 0-100", "High Res",
                "HighRes SmallVess", "Point Illumination", "Skin", "Acoustic"]
    for dataset in datasets:
        train_all(dataset, n_spectra=50000, flowphantom=True)
        train_all(dataset, n_spectra=50000)
        # train_all(dataset, n_spectra=77000)

    print("--- %s seconds ---" % (time.time() - start_time))