import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors

temp = 'C:/Users\yeung01\PycharmProjects\qpai-training\kylie\KylieBaseline_spectra.hdf5'
f_temp = h5py.File(temp,'r')
NUM_SIMULATIONS = 5
WAVELENGTHS = np.linspace(700,900,41)

spectra_a = np.zeros((42,1))
spectra_a[1:42, 0] = WAVELENGTHS.T
spectra_b = np.copy(spectra_a)
spectra_c = np.copy(spectra_a)
for set in range(1):
    print(set)
    data = np.array(f_temp[f'Set {set}'])
    print(data.shape)
    # try:
    spectraset_a = data[:, np.where(data[0,:] <= 1/3)]  # bin the spectra according to level of oxygenation
    print(f"spectraset_a shape {spectraset_a.shape}")
    spectra_a = np.vstack((spectra_a,spectraset_a))
    print(f"spectra_a shape {spectra_a.shape}")
    # except IndexError:
    #     print(f"No vessels of 0-33% oxy in set {set}.")
    # try:
    #     spectraset_b = data[np.where((data[0,:] > 1/3) & (data[0,:] <= 2/3)), :]  # bin the spectra according to level of oxygenation
    #     print(f"spectraset_b shape {spectraset_b.shape}")
    #     spectra_b = np.vstack((spectra_b,spectraset_b))
    #     print(f"spectra_b shape {spectra_b.shape}")
    # except IndexError:
    #     print(f"No vessels of 33-66% oxy in set {set}.")
    # try:
    #     spectraset_c = data[np.where(data[0,:] > 2/3), :]  # bin the spectra according to level of oxygenation
    #     print(f"spectraset_c shape {spectraset_c.shape}")
    #     spectra_c = np.vstack((spectra_c,spectraset_c))
    #     print(f"spectra_c shape {spectra_c.shape}")
    # except IndexError:
    #     print(f"No vessels of 66-100% oxy in set {set}.")


def normalise(x):
    return x / np.sum(x)

# x = WAVELENGTHS
# ys = spectra_a[1:, 1:]
# ys = np.apply_along_axis(normalise, axis=1, arr=ys)
# ax = plt.axes()
# ax.set_xlim(x.min(), x.max())
# ax.set_ylim(ys.min(), ys.max())
# line_segments = LineCollection([np.column_stack([x, y]) for y in ys])
#
# line_segments.set_array(spectra_a[0, 1:, 0])  # this line color codes by oxy (?)
# ax.add_collection(line_segments)
# fig = plt.gcf()  # get current figure
# ax.add_collection(line_segments)
# axcb = fig.colorbar(line_segments)
# axcb.set_label('oxygenation')
# ax.set_title('Line Collection with mapped colors')
# plt.sci(line_segments)  # This allows interactive changing of the colormap.
# plt.show()

# fig, (ax1, ax2, ax3) = plt.subplots(1,3)
# # plot disregard first row (wavelengths) and first column (oxy)
#
# ax1.plot(WAVELENGTHS, np.transpose(spectra_a[0, 1:, 1:]), alpha=0.05, color=spectra_a[0,1:,0])
# ax2.plot(WAVELENGTHS, np.transpose(spectra_b[0, 1:, 1:]), alpha=0.05, color='g')
# ax3.plot(WAVELENGTHS, np.transpose(spectra_c[0, 1:, 1:]), alpha=0.05, color='b')
#
# plt.show()
