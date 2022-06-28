from simpa.utils import Tags
import simpa as sp
import numpy as np
from simpa.io_handling import load_hdf5
from simpa.utils import get_data_field_from_simpa_output

path_manager = sp.PathManager()
VOLUME_NAME= "OxygenationSimulation_100000"
file=path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5"
WAVELENGTHS = [800, 805]

# initial_pressure_mat = np.zeros()  # create a matrix in which to store the initial pressures by wavelength
wavelength = WAVELENGTHS[0]
sp.visualise_data(path_to_hdf5_file=file,
                      wavelength=wavelength,
                      show_initial_pressure=True,
                      show_absorption=True,
                      log_scale=True)
# obtaining initial pressure (copied from visualisation)
file = load_hdf5(file)

initial_pressure = get_data_field_from_simpa_output(file, Tags.DATA_FIELD_INITIAL_PRESSURE,
                                                                          wavelength)
