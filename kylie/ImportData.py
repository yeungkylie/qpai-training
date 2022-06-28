import simpa as sp
import h5py
import numpy as np

path_manager = sp.PathManager()
VOLUME_NAME= "KylieBaseline_"
temp = path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + str(int(1e5)) + ".hdf5"
WAVELENGTHS = [800, 805, 810]

f_temp = h5py.File(temp,'r')
temp_seg_array = np.array(f_temp['simulations']['simulation_properties']['seg'])  # trying to segment out the vessels
temp_oxygenation_array = f_temp['simulations']['simulation_properties']['oxy']
i,j,k = np.where(temp_seg_array == 3.0)
vessel_indices = [[i[n],j[n],k[n]] for n in range(len(i))]

f_store = h5py.File("KylieBaseline_spectra.hdf5", "w")  # created dataset

for n in range(int(1e5), int(1e5)+3):  # num simulations
    print(n)
    filename = path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + str(n) + '.hdf5'

    f = h5py.File(filename,'r')
    oxygenation_array = f['simulations']['simulation_properties']['oxy']
    oxygenations = [oxygenation_array[entry[0]][entry[1]][entry[2]] for entry in vessel_indices]
    data = f['simulations']['optical_forward_model_output']['initial_pressure']

    initial_pressures = [[] for n in range(len(vessel_indices))]  # change variables(?)

    for k in range(2):
        wavelength = str(800 + k*5)
        wavelength_data = data[wavelength]
        wavelength_pressures = [wavelength_data[entry[0]][entry[1]][entry[2]] for entry in vessel_indices]
        for l in range(len(initial_pressures)):
            initial_pressures[l].append(wavelength_pressures[l])

    final_dset = [[oxygenations[m]]+initial_pressures[m] for m in range(len(vessel_indices))]
    dset = f_store.create_dataset('Set ' + "{0}".format(n), data=final_dset)