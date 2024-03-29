from simpa.utils import Tags
import simpa as sp
import numpy as np
import utils.plot_vessel_tree as plt_vessels

# FIXME temporary workaround for newest Intel architectures
import os
import time
start_time = time.time()  # calculate runtime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

VOLUME_TRANSDUCER_DIM_IN_MM = 19.2  # 64 pixels
VOLUME_PLANAR_DIM_IN_MM = 19.2
VOLUME_HEIGHT_IN_MM = 19.2
SPACING = 0.3
NUM_VERTICAL_COMPARTMENTS = 3
NUM_HORIZONTAL_COMPARTMENTS = 2
# WAVELENGTHS = np.linspace(700, 900, 41, dtype=int)  # full 41 wavelengths
WAVELENGTHS = [800]
NUM_SIMULATIONS = 1

path_manager = sp.PathManager()

# If VISUALIZE is set to True, the simulation result will be plotted
VISUALIZE = True

def create_example_tissue():
    """
    Tissue definition containing muscular background and 0-3 blood vessels.
    """
    background_dictionary = sp.Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-10, 1e-10, 1.0)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary

    # constant bvf and oxy in baseline simulation
    blood_volume_fraction = 0.01
    oxy = 0.7
    tissue_dict["muscle"] = sp.define_horizontal_layer_structure_settings(z_start_mm=0, thickness_mm=100,
                                                                          molecular_composition=
                                                                          sp.TISSUE_LIBRARY.muscle(
                                                                              oxy, blood_volume_fraction),
                                                                          priority=1,
                                                                          consider_partial_volume=True,
                                                                          adhere_to_deformation=False)

    for vertical_idx in range(NUM_VERTICAL_COMPARTMENTS):
        for horizontal_idx in range(NUM_HORIZONTAL_COMPARTMENTS):
            idx = vertical_idx * NUM_HORIZONTAL_COMPARTMENTS + horizontal_idx
            # Randomising whether this compartment contains a vessel
            vessel_probability = 1
            vessel_randomisation = np.random.random()
            if vessel_randomisation < vessel_probability:
                # randomise the radius to be somewhere between e.g. 0.3 and 2 mm
                lower_tube_radius = 0.5
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
                    # Draw another random variable for the oxygen saturation for each vessel between 0 and 1
                vessel_oxy_sat = np.random.random()
                start_y = np.random.randint(2)*VOLUME_PLANAR_DIM_IN_MM  # start from either end of y
                tissue_dict[f"vessel_{idx}"] = sp.define_vessel_structure_settings(
                    vessel_start_mm=[start_x, start_y, start_z],
                    vessel_direction_mm=[0, abs(VOLUME_PLANAR_DIM_IN_MM - start_y), 0],  # make vessels grow towards the opposite direction
                    molecular_composition=sp.TISSUE_LIBRARY.blood(oxygenation=vessel_oxy_sat),
                    radius_mm=tube_radius,
                    curvature_factor=0.05, #0.05
                    radius_variation_factor=1,  # 1
                    bifurcation_length_mm=2,  # 7
                    priority=3, consider_partial_volume=True,
                    adhere_to_deformation=False
                )
        plt_vessels.plot_vessel_tree(settings, tissue_dict,
                                     NUM_HORIZONTAL_COMPARTMENTS * NUM_VERTICAL_COMPARTMENTS)  # storing the vessel shape for ref
    return tissue_dict


# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume is generated with the same random seed every time.

for simulation_idx in range(NUM_SIMULATIONS):
    # Every volume needs a distinct random seed.
    RANDOM_SEED = int(1e4 + simulation_idx)

    np.random.seed(RANDOM_SEED)
    VOLUME_NAME = "KylieBaseline_" + str(RANDOM_SEED)

    general_settings = {
        # These parameters set the general properties of the simulated volume
        Tags.RANDOM_SEED: RANDOM_SEED,
        Tags.VOLUME_NAME: VOLUME_NAME,
        Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
        Tags.SPACING_MM: SPACING,
        Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
        Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
        Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
        Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,
        Tags.GPU: True,
        Tags.WAVELENGTHS: WAVELENGTHS,
        Tags.DO_FILE_COMPRESSION: True,
        Tags.DO_IPASC_EXPORT: False  # since there is no acoustic modelling
    }
    settings = sp.Settings(general_settings)

    settings.set_volume_creation_settings({
        Tags.STRUCTURES: create_example_tissue(),
        Tags.SIMULATE_DEFORMED_LAYERS: True
    })

    settings.set_optical_settings({
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
        Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
        Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_GAUSSIAN,
        Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
        Tags.MCX_ASSUMED_ANISOTROPY: 0.9,
    })

    device = sp.PhotoacousticDevice(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                                                 VOLUME_PLANAR_DIM_IN_MM / 2,
                                                                 0]))
    device.add_illumination_geometry(sp.GaussianBeamIlluminationGeometry(beam_radius_mm=12))

    SIMULATION_PIPELINE = [
        sp.ModelBasedVolumeCreationAdapter(settings),
        sp.MCXAdapter(settings),
    ]

    sp.simulate(SIMULATION_PIPELINE, settings, device)

    sp.visualise_data(path_to_hdf5_file=path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5",
                      wavelength=WAVELENGTHS[0],
                      show_initial_pressure=True,
                      show_absorption=True,
                      log_scale=True,
                      show_xz_only=False)

    print("--- %s seconds ---" % (time.time() - start_time))