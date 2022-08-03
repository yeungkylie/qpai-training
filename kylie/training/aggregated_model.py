import pickle
import os
import numpy as np
from sklearn.metrics import median_absolute_error
from kylie.post_processing import prepare_simpa_simulations as p


def aggregated_model(SET_NAME, process, n_training_spectra, validation_spectra, gt_oxy):
    rfr = []
    pred = []
    MODEL_FOLDER = f"I:/research\seblab\data\group_folders\Kylie\Trained Models {n_training_spectra}/{SET_NAME}/{process}"  # where the models are stored
    for idx in range(5):  # for each fold
        rfr_idx = pickle.load(open(os.path.join(MODEL_FOLDER, f"{SET_NAME}_{process}_rf{idx}.sav"), 'rb'))
        rfr.append(rfr_idx)
        pred_idx = rfr[idx].predict(validation_spectra)  # predict oxygenation using each fold
        pred.append(pred_idx)
        print(f"model{idx} prediction: {pred_idx}")
    final_prediction = np.mean(pred, axis=0)  # take the mean of the predictions as final
    print(f"final prediction: {final_prediction}")
    ae = np.abs(final_prediction - gt_oxy)  # store absolute error for all predictions
    mae = median_absolute_error(gt_oxy, final_prediction)
    np.savez(f"I:/research\seblab\data\group_folders\Kylie/validation/{SET_NAME}_{n_training_spectra}_{process}_validation.npz",
             ground_truth=gt_oxy, prediction=final_prediction, ae=ae, mae=mae)
    print(f"Median absolute error {mae}")


def validate_all(validation_spectra, ground_truth):
    datasets = ["Baseline", "0.6mm Res", "1.2mm Res", "5mm Illumination",
                "Point Illumination", "BG 0-100", "BG 60-80",
                "Heterogeneous with vessels", "High Res",
                "HighRes SmallVess", "Point Illumination", "Skin"]
    processes = [None, "thresholded", "smoothed", "noised", "thresholded_smoothed"]

    for dataset in datasets:
        for process in processes:
            aggregated_model(dataset, process, 77000, validation_spectra, ground_truth)
            aggregated_model(dataset, process, 400000, validation_spectra, ground_truth)


if __name__ == "__main__":
    # example validating using "test extraction" spectra
    IN_FILE = f"I:/research\seblab\data\group_folders\Kylie/datasets/test extraction/test extraction_spectra.npz"
    r_wavelengths, r_oxygenations, r_spectra, \
    r_melanin_concentration, r_background_oxygenation, \
    r_distances, r_depths, r_pca_components = p.load_spectra_file(IN_FILE)

    validation_spectra = r_spectra.T  # may have to transpose to fit dimensions
    ground_truth = r_oxygenations

    validate_all(validation_spectra, ground_truth)