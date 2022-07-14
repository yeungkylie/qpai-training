import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
import prepare_simpa_simulations

SET_NAME = "Baseline"
IN_PATH = f"D:/Kylie Simulations/{SET_NAME}/"
OUT_FILE = f"D:/Kylie Simulations/datasets/{SET_NAME}/{SET_NAME}_spectra.npz"

r_wavelengths, r_oxygenations, r_spectra, \
r_melanin_concentration, r_background_oxygenation, \
r_distances, r_depths = prepare_simpa_simulations.load_spectra_file(OUT_FILE)


pca = PCA(n_components=2)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
components = pca.fit_transform(np.array(r_spectra))

fig = px.scatter(components, x=0, y=1)
fig.show()