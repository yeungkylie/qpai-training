from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from kylie import prepare_simpa_simulations as p
import numpy as np
import pickle

SET_NAME = "test extraction"
IN_FILE = f"D:/Kylie Simulations/datasets/{SET_NAME}/{SET_NAME}_spectra.npz"
OUT_FILE = f"I:/research\seblab\data\group_folders\Kylie\Trained Models/{SET_NAME}.sav"
rfr = pickle.load(open(OUT_FILE, 'rb'))

r_wavelengths, r_oxygenations, r_spectra, \
r_melanin_concentration, r_background_oxygenation, \
r_distances, r_depths, r_pca_components = p.load_spectra_file(IN_FILE)

x, y = r_spectra.T, r_oxygenations  # x is the training data and y is the target
X = scale(x)
Y = scale(y)

scores = []
kf = KFold(n_splits=5, shuffle=False)
for train_index, test_index in kf.split(X):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)
    xtrain, xtest, ytrain, ytest = X[train_index], X[test_index], y[train_index], y[test_index]
    scores.append(rfr.score(xtrain, ytrain))

print(scores)
print(f"Averaged score: {np.mean(scores)}")
