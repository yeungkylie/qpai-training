import h5py
import numpy as np

temp = 'D:/tom_sims/data_kevin/Flowphantom_inVision_700_900_5_Water_   0.hdf5'
f_temp = h5py.File(temp,'r')
temp_seg_array = np.array(f_temp['simulations']['simulation_properties']['seg'])
temp_oxygenation_array = f_temp['simulations']['simulation_properties']['oxy']
i,j,k = np.where(temp_seg_array == 3.0)
vessel_indices = [[i[n],j[n],k[n]] for n in range(len(i))]


f1 = h5py.File("flowphantom_complicated.hdf5", "w")

for i in range(601):
    print(i)
    if i < 10:
        filename = 'D:/tom_sims/data_kevin/Flowphantom_inVision_700_900_5_Water_   ' + str(i) + '.hdf5'
    elif i < 100:
        filename = 'D:/tom_sims/data_kevin/Flowphantom_inVision_700_900_5_Water_  ' + str(i) + '.hdf5'
    else:
        filename = 'D:/tom_sims/data_kevin/Flowphantom_inVision_700_900_5_Water_ ' + str(i) + '.hdf5'

    f = h5py.File(filename,'r')
    oxygenation_array = f['simulations']['simulation_properties']['oxy']
    oxygenations = [oxygenation_array[entry[0]][entry[1]][entry[2]] for entry in vessel_indices]
    data = f['simulations']['optical_forward_model_output']['initial_pressure']

    initial_pressures = [[] for i in range(len(vessel_indices))]

    for k in range(41):
        wavelength = str(700 + k*5)
        wavelength_data = data[wavelength]
        wavelength_pressures = [wavelength_data[entry[0]][entry[1]][entry[2]] for entry in vessel_indices]
        for l in range(len(initial_pressures)):
            initial_pressures[l].append(wavelength_pressures[l])

    final_dset = [[oxygenations[k]]+initial_pressures[k] for k in range(len(vessel_indices))]
    dset = f1.create_dataset('Set ' + "{0}".format(i), data=final_dset)