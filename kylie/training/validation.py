import pickle
import os
import numpy as np
from sklearn.metrics import median_absolute_error
from kylie.post_processing import prepare_simpa_simulations as p


def aggregated_model(SET_NAME, process, n_training_spectra, test_data, flowphantom=True):
    print(f"Loading test data: {test_data} ...")
    IN_FILE = f"I:/research\seblab\data\group_folders\Janek\learned_pa_oximetry/validation_data\in_silico/{test_data}/{test_data}.npz"
    r_wavelengths, r_oxygenations, r_spectra, \
    r_melanin_concentration, r_background_oxygenation, \
    r_distances, r_depths = p.load_spectra_file(IN_FILE)

    validation_spectra = r_spectra  # may have to transpose to fit dimensions
    gt_oxy = r_oxygenations

    rfr = []
    pred = []
    if not flowphantom:
        MODEL_FOLDER = f"I:/research\seblab\data\group_folders\Kylie\Trained Models {n_training_spectra}/{SET_NAME}/{process}"  # where the models are stored
    else:
        MODEL_FOLDER = f"I:/research\seblab\data\group_folders\Kylie\Trained Models {n_training_spectra} flowphantom/{SET_NAME}/{process}"
        validation_spectra = (filter_wavelengths(validation_spectra)).T
    print(f"Loading model from {MODEL_FOLDER}...")
    print(validation_spectra.shape)
    for idx in range(5):  # for each fold
        rfr_idx = pickle.load(open(os.path.join(MODEL_FOLDER, f"{SET_NAME}_{process}_rf{idx}.sav"), 'rb'))
        rfr.append(rfr_idx)
        pred_idx = rfr[idx].predict(validation_spectra)  # predict oxygenation using each fold
        pred.append(pred_idx)
        print(f"Fold {idx} prediction: {pred_idx}")
    final_prediction = np.mean(pred, axis=0)  # take the mean of the predictions as final
    print(f"Final prediction: {final_prediction}")
    ae = np.abs(final_prediction - gt_oxy)  # store absolute error for all predictions
    mae = median_absolute_error(gt_oxy, final_prediction)
    OUT_FOLDER = f"I:/research\seblab\data\group_folders\Kylie/validation/{test_data}/"
    OUT_FILE = f"{SET_NAME}_{n_training_spectra}_{process}_validation.npz"
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)
    np.savez(os.path.join(OUT_FOLDER,OUT_FILE),
             ground_truth=gt_oxy, prediction=final_prediction, ae=ae, mae=mae)
    print(f"Median absolute error {mae}")

def filter_wavelengths(spectra):
    flowphantom_wavelengths = np.asarray([700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900])
    wavelength_idx = (flowphantom_wavelengths - 700) / 5
    r_spectra = spectra[wavelength_idx.astype(int), :]
    return r_spectra

def validate_all(test_data):
    datasets = ["Baseline", "0.6mm Res", "1.2mm Res", "5mm Illumination",
                "Point Illumination", "BG 0-100", "BG 60-80",
                "Heterogeneous with vessels", "Heterogeneous 60-80",
                "Heterogeneous 0-100", "High Res",
                "HighRes SmallVess", "Skin"]
    processes = [None, "thresholded", "smoothed", "noised", "thresholded_smoothed"]

    for dataset in datasets:
        for process in processes:
            print(f"Testing on {dataset} {process} model...")
            aggregated_model(dataset, process, 50000, test_data)


if __name__ == "__main__":
    in_silico = ["Simulation1_SingleVesselInWater",
                 "Simulation2_SingleVesselInBlood",
                 "Simulation3_VesselDeepInWater",
                 "Simulation4_HeterogeneousDistribution"]
    for test_data in in_silico:
          validate_all(test_data)