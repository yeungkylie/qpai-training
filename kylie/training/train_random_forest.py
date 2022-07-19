from sklearn.ensemble import RandomForestRegressor
from kylie import prepare_simpa_simulations as p
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold
import pickle
import time
start_time = time.time()

def get_spectra_oxy(SET_NAME):
    IN_FILE = f"D:/Kylie Simulations/datasets/{SET_NAME}/{SET_NAME}_spectra.npz"
    r_wavelengths, r_oxygenations, r_spectra, \
    r_melanin_concentration, r_background_oxygenation, \
    r_distances, r_depths, r_pca_components = p.load_spectra_file(IN_FILE)
    spectra = scale(r_spectra.T)
    # spectra = scale(r_spectra)  # fast inaccurate check
    oxy = scale(r_oxygenations)
    return spectra, oxy

def train_random_forests(SET_NAME):
    spectra, oxy = get_spectra_oxy(SET_NAME)
    scores = []
    mse = []
    kf = KFold(n_splits=5, shuffle=False)
    for idx, (train_index, test_index) in enumerate(kf.split(spectra)):
        print("Train Index: ", train_index, "\n")
        print("Test Index: ", test_index)
        xtrain, xtest, ytrain, ytest = spectra[train_index], spectra[test_index], oxy[train_index], oxy[test_index]
        rfr = RandomForestRegressor(n_estimators=70, verbose=2, max_depth=16, n_jobs=-1)
        rfr.fit(xtrain, ytrain)

        scores.append(rfr.score(xtrain, ytrain))
        print(scores)  # shows how closely the target oxy can be regressed to spectra

        ypred = rfr.predict(xtest)

        mse.append(mean_squared_error(ytest, ypred))
        print("MSE: ", mse)
        # print("RMSE: ", mse*(1/2.0))


        # x_ax = range(len(ytest))
        # plt.plot(x_ax, ytest, label="original")
        # plt.plot(x_ax, ypred, label="predicted")
        # plt.xlabel('spectra-axis')
        # plt.ylabel('Y-axis')
        # plt.legend(loc='best',fancybox=True, shadow=True)
        # plt.grid(True)
        # plt.show()

        OUT_FOLDER = f"I:/research\seblab\data\group_folders\Kylie\Trained Models/"
        OUT_FILE = f"{SET_NAME}_rf{idx}.sav"
        pickle.dump(rfr, open(os.path.join(OUT_FOLDER, OUT_FILE), 'wb'))  # save each fold
    mean_score = np.mean(scores)
    mean_mse = np.mean(mse)
    print(f"Averaged score: {np.mean(scores)}")  # metric combining r-score of all folds
    print(f"Averaged MSE: {np.mean(mse)}")  # metric combining mse of all folds
    return mean_score, mean_mse


def aggregated_model(SET_NAME, input_spectra, target_oxy=None):
    rfr = []
    ypred = []
    OUT_FOLDER = f"I:/research\seblab\data\group_folders\Kylie\Trained Models/"  # where the models are stored
    for idx in range(5):  # for each fold
        rfr_idx = pickle.load(open(os.path.join(OUT_FOLDER, f"{SET_NAME}_rf{idx}.sav"), 'rb'))
        rfr.append(rfr_idx)
        ypred_idx = rfr[idx].predict(input_spectra)
        ypred.append(ypred_idx)
        print(f"model{idx} prediction: {ypred[idx]}")
    final_prediction = np.mean(ypred, axis=0)
    print(f"final prediction: {final_prediction}")
    if target_oxy is not None:
        mse = mean_squared_error(target_oxy, final_prediction)
        print(f"MSE {mse}")


if __name__ == "__main__":
    train_random_forests("Baseline")
    # aggregated_model(spectra[0:6], oxy[0:6])
    print("--- %s seconds ---" % (time.time() - start_time))