# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa import Tags
import simpa as sp
import numpy as np
import matplotlib.pyplot as plt
from utils.save_directory import get_save_path
import os

def plot_vessel_tree(settings, tissue_dict, total_vessel_num):
    SAVE_PATH = get_save_path("Tissue_Generation", "Vessel_Tree")

    fontsize = 13
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    print("Plotting and saving vessel trees...")
    try:
        for idx in range(total_vessel_num):
            if tissue_dict[f"vessel_{idx}"]:  # skip plotting if the vessel does not exist in certain compartment
                settings.set_volume_creation_settings({
                    Tags.STRUCTURES: tissue_dict[f"vessel_{idx}"],
                    Tags.SIMULATE_DEFORMED_LAYERS: True
                })
                vessel = sp.VesselStructure(settings, tissue_dict[f"vessel_{idx}"])
                vessel_tree = vessel.geometrical_volume
                ax.voxels(vessel_tree, shade=True, alpha=0.55)
    except KeyError:
        print(f"No vessel in compartment {idx}")


    ax.set_aspect('auto')
    ax.set_xticks(np.linspace(0, settings[Tags.DIM_VOLUME_X_MM]/settings[Tags.SPACING_MM], 6))
    ax.set_yticks(np.linspace(0, settings[Tags.DIM_VOLUME_Y_MM]/settings[Tags.SPACING_MM], 6))
    ax.set_zticks(np.linspace(0, settings[Tags.DIM_VOLUME_Z_MM]/settings[Tags.SPACING_MM], 6))
    ax.set_xticklabels(np.linspace(0, settings[Tags.DIM_VOLUME_X_MM], 6, dtype=int), fontsize=fontsize)
    ax.set_yticklabels(np.linspace(0, settings[Tags.DIM_VOLUME_Y_MM], 6, dtype=int), fontsize=fontsize)
    ax.set_zticklabels(np.linspace(0, settings[Tags.DIM_VOLUME_Z_MM], 6, dtype=int), fontsize=fontsize)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_zlim(int(settings[Tags.DIM_VOLUME_X_MM]/settings[Tags.SPACING_MM]), 0)
    ax.set_zlabel("Depth [mm]", fontsize=fontsize)
    ax.set_xlabel("x width [mm]", fontsize=fontsize)
    ax.set_ylabel("y width [mm]", fontsize=fontsize)
    # plt.axis("off")
    ax.view_init(elev=10., azim=-45)
    plt.savefig(os.path.join(SAVE_PATH, f"vessel_tree_{total_vessel_num}.svg"), dpi=300)
    plt.close()