import spectra_extraction_no_vessel as senv
from kylie import postprocessing as pp
from kylie.training import train_random_forest as trf

senv.extract_spectra("Heterogeneous 0-100")
pp.generate_processed_datasets("Heterogeneous 0-100")
trf.train_all("Heterogeneous 0-100", n_spectra=400000)
trf.train_all("Heterogeneous 0-100", n_spectra=77000)
