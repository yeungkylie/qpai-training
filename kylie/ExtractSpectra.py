import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
'''HIGH MELANIN RUN'''

f = h5py.File('I:/research/seblab/data/group_folders/Tom/Melanin_Flow Phantom/Processed_data/Scan_5.hdf5')
data = f['recons']['OpenCL Backprojection']['0']
timesteps = len(data)
wavelengths = [700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900]
wavelength_indices = [2,3,4,5,6,7,8,9,10,11,12]
chosen_pixels = [[155,169]]  # manual segmentation
chosen_pixels += [[156,i] for i in range(166,173)]
chosen_pixels += [[157,i] for i in range(164, 175)]
chosen_pixels += [[158,i] for i in range(163, 176)]
chosen_pixels += [[159,i] for i in range(163, 176)]
chosen_pixels += [[160,i] for i in range(162, 177)]
chosen_pixels += [[161,i] for i in range(162, 177)]
chosen_pixels += [[162,i] for i in range(162, 177)]
chosen_pixels += [[163,i] for i in range(161, 178)]
chosen_pixels += [[164,i] for i in range(162, 177)]
chosen_pixels += [[165,i] for i in range(162, 177)]
chosen_pixels += [[166,i] for i in range(162, 177)]
chosen_pixels += [[167,i] for i in range(163, 176)]
chosen_pixels += [[168,i] for i in range(163, 176)]
chosen_pixels += [[169,i] for i in range(164, 175)]
chosen_pixels += [[170,i] for i in range(166,173)]
chosen_pixels += [[171,169]]

for i in range(timesteps):
    timestep_data = data[i]
    timestep_spectra = [[] for i in range(len(chosen_pixels))]
    for index in wavelength_indices:
        for j in range(len(chosen_pixels)):
            timestep_spectra[j].append(timestep_data[index][chosen_pixels[j][1]][chosen_pixels[j][0]][0])
    timestep_spectra = torch.tensor(timestep_spectra)
    filename = 'Timestep' + str(i) + '.pt'
    torch.save(timestep_spectra, filename)
