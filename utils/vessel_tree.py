# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa import Tags
import simpa as sp
import matplotlib.pyplot as plt
import numpy as np
from utils.save_directory import get_save_path
import os

SAVE_PATH = get_save_path("Tissue_Generation", "Vessel_Tree")
np.random.seed(1234)

VOLUME_TRANSDUCER_DIM_IN_MM = 19.2  # 64 pixels
VOLUME_PLANAR_DIM_IN_MM = 19.2
VOLUME_HEIGHT_IN_MM = 9.6
SPACING = 0.3
NUM_VERTICAL_COMPARTMENTS = 2
NUM_HORIZONTAL_COMPARTMENTS = 1

global_settings = sp.Settings()
global_settings[Tags.SPACING_MM] = SPACING
global_settings[Tags.DIM_VOLUME_X_MM] = VOLUME_TRANSDUCER_DIM_IN_MM
global_settings[Tags.DIM_VOLUME_Y_MM] = VOLUME_PLANAR_DIM_IN_MM
global_settings[Tags.DIM_VOLUME_Z_MM] = VOLUME_HEIGHT_IN_MM
global_settings.set_volume_creation_settings({
    Tags.SIMULATE_DEFORMED_LAYERS: False,
})

tissue_dict = sp.Settings()
settings = sp.Settings(global_settings)

for vertical_idx in range(NUM_VERTICAL_COMPARTMENTS):
    for horizontal_idx in range(NUM_HORIZONTAL_COMPARTMENTS):
        idx = vertical_idx * NUM_HORIZONTAL_COMPARTMENTS + horizontal_idx
        # Randomising whether this compartment contains a vessel
        vessel_probability = 1
        vessel_randomisation = np.random.random()
        if vessel_randomisation < vessel_probability:
            # randomise the radius to be somewhere between e.g. 0.3 and 2 mm
            lower_tube_radius = 0.3
            upper_tube_radius = 2
            tube_radius = (lower_tube_radius - upper_tube_radius) * np.random.random() + upper_tube_radius
            # Define min and max of x and z based on VOLUME_TRANSDUCER_DIM_IN_MM (x)
            # and VOLUME_HEIGHT_IN_MM (z), ensuring no overlap and no out of bounds
            max_x = (VOLUME_TRANSDUCER_DIM_IN_MM * (horizontal_idx + 1) / NUM_HORIZONTAL_COMPARTMENTS) - tube_radius
            max_z = (VOLUME_HEIGHT_IN_MM * (vertical_idx + 1) / NUM_VERTICAL_COMPARTMENTS) - tube_radius
            min_x = (max_x - VOLUME_TRANSDUCER_DIM_IN_MM / NUM_HORIZONTAL_COMPARTMENTS) + 2*tube_radius
            min_z = (max_z - VOLUME_HEIGHT_IN_MM / NUM_VERTICAL_COMPARTMENTS) + 2*tube_radius
            # Then create two random variables for the x and z position. The y position is already set correctly
            start_x, end_x = (min_x - max_x) * np.random.random(2) + max_x
            start_z, end_z = (min_z - max_z) * np.random.random(2) + max_z
            vessel_oxy_sat = np.random.random()
            vessel_settings = sp.define_vessel_structure_settings(
                vessel_start_mm=[start_x, 0, start_z],
                vessel_direction_mm=[0,1,0],
                molecular_composition=sp.TISSUE_LIBRARY.blood(oxygenation=vessel_oxy_sat),
                radius_mm=tube_radius,
                curvature_factor=0.05,# 0.05
                radius_variation_factor=0.5,  # 1
                bifurcation_length_mm=7,  # 7
                priority=3, consider_partial_volume=True,
                adhere_to_deformation=False)
            vessel = sp.VesselStructure(global_settings, vessel_settings)
            vessel_tree = vessel.geometrical_volume

print("Plotting...")
fontsize = 13
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.voxels(vessel_tree, shade=True, facecolors="red", alpha=0.55)
ax.set_aspect('auto')
# ax.set_xticks(np.linspace(0, global_settings[Tags.DIM_VOLUME_X_MM]/global_settings[Tags.SPACING_MM], 6))
# ax.set_yticks(np.linspace(0, global_settings[Tags.DIM_VOLUME_Y_MM]/global_settings[Tags.SPACING_MM], 6))
# ax.set_zticks(np.linspace(0, global_settings[Tags.DIM_VOLUME_Z_MM]/global_settings[Tags.SPACING_MM], 6))
# ax.set_xticklabels(np.linspace(0, global_settings[Tags.DIM_VOLUME_X_MM], 6, dtype=int), fontsize=fontsize)
# ax.set_yticklabels(np.linspace(0, global_settings[Tags.DIM_VOLUME_X_MM], 6, dtype=int), fontsize=fontsize)
# ax.set_zticklabels(np.linspace(0, global_settings[Tags.DIM_VOLUME_X_MM], 6, dtype=int), fontsize=fontsize)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_zlim(int(global_settings[Tags.DIM_VOLUME_X_MM]/global_settings[Tags.SPACING_MM]), 0)
# ax.set_zlabel("Depth [mm]", fontsize=fontsize)
# ax.set_xlabel("x width [mm]", fontsize=fontsize)
# ax.set_ylabel("y width [mm]", fontsize=fontsize)
# plt.axis("off")
ax.view_init(elev=10., azim=-45)
plt.savefig(os.path.join(SAVE_PATH, "vessel_tree.svg"), dpi=300)
plt.close()