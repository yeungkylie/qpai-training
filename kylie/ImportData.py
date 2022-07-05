import simpa as sp
import h5py
import numpy as np
import time
start_time = time.time()

path_manager = sp.PathManager()
VOLUME_NAME= "KylieBaseline_"
WAVELENGTHS = np.linspace(700, 900, 41, dtype=int)  # full 41 wavelengths
# WAVELENGTHS = [800]
NUM_SIMULATIONS = 500

f_store = h5py.File(VOLUME_NAME + "spectra.hdf5", "w")  # create spectra dataset

for n in range(NUM_SIMULATIONS):  # num simulations
    print(n)
    filename = path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + str(int(1e4 + n)) + '.hdf5'
    f = h5py.File(filename,'r')

    # getting segmentation of vessels
    seg_array = np.array(f['simulations']['simulation_properties']['seg'])
    i, j, k = np.where(seg_array == 3.0)  # priority tag assigned to vessels in sim
    vessel_coordinates = [[i[m], j[m], k[m]] for m in range(len(i))]  # rearrange to sets of 3d cooordinates

    # find oxy in each vessel voxel
    oxygenation_array = f['simulations']['simulation_properties']['oxy']  # of volume dimensions

    oxygenations = [oxygenation_array[c[0]][c[1]][c[2]] for c in vessel_coordinates]
    # find spectra in each vessel voxel
    p_o = f['simulations']['optical_forward_model_output']['initial_pressure']
    initial_pressures = [[] for n in range(len(vessel_coordinates))]  # change variables(?)
    for k in range(41):  # for every wavelength simulated
        wavelength = str(700 + k*5)
        wavelength_data = p_o[wavelength]  # initial pressure at given wavelength
        wavelength_pressures = [wavelength_data[c[0]][c[1]][c[2]] for c in vessel_coordinates]
        for l in range(len(initial_pressures)):
            initial_pressures[l].append(wavelength_pressures[l])

    # create and store dataset (rows: wavelengths; column 0: oxy, column 1-41: p_o)
    final_dset = [[oxygenations[m]]+initial_pressures[m] for m in range(len(vessel_coordinates))]
    dset = f_store.create_dataset('Set ' + "{0}".format(n), data=final_dset)

print("--- %s seconds ---" % (time.time() - start_time))