import simpa as sp
import h5py
import numpy as np
import time
start_time = time.time()

path_manager = sp.PathManager()
VOLUME_NAME= "KylieBaseline_"
WAVELENGTHS = np.linspace(700, 900, 41, dtype=int)  # full 41 wavelengths
# WAVELENGTHS = [800]
NUM_SIMULATIONS = 20

f_store = h5py.File(VOLUME_NAME + "spectra.hdf5", "w")  # create spectra dataset

for n in range(NUM_SIMULATIONS):  # num simulations
    print(f"Set {n}")
    # filename = path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + str(int(1e4 + n)) + '.hdf5'
    filename = "I:/research/seblab/data/group_folders/Kylie/Baseline/" + VOLUME_NAME + str(int(1e4 + n)) + '.hdf5'
    f = h5py.File(filename,'r')

    # getting segmentation of vessels
    seg_array = np.array(f['simulations']['simulation_properties']['seg'])
    # mask_init = np.ones_like(seg_array)
    # mask = mask_init*(seg_array==3)
    # print(mask)
    NUM_SPECTRA = np.count_nonzero(seg_array == 3)  # number of spectra extracted from vessels

    # find oxy in each vessel voxel
    oxygenation_array = f['simulations']['simulation_properties']['oxy']  # of volume dimensions
    oxygenations = oxygenation_array[seg_array==3]
    final_dset = np.empty((42, NUM_SPECTRA))  # create empty array for storing spectra
    final_dset[0,:] = oxygenations

    # find spectra in each vessel voxel
    p_o = f['simulations']['optical_forward_model_output']['initial_pressure']

    for k in range(41):  # for every wavelength simulated
        print(f"Wavelength {700+k*5}nm")
        wavelength = str(700 + k*5)
        wavelength_data = p_o[wavelength][seg_array==3]  # initial pressure at given wavelength
        final_dset[k+1, :] = wavelength_data

    print(final_dset)
    dset = f_store.create_dataset('Set ' + "{0}".format(n), data=final_dset)

print("--- %s seconds ---" % (time.time() - start_time))