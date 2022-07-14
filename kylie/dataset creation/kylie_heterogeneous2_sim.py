from simpa.utils import Tags
import simpa as sp
import numpy as np

# FIXME temporary workaround for newest Intel architectures
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

VOLUME_TRANSDUCER_DIM_IN_MM = 19.2  # 64 pixels
VOLUME_PLANAR_DIM_IN_MM = 19.2
VOLUME_HEIGHT_IN_MM = 19.2
SPACING = 0.3
NUM_VERTICAL_COMPARTMENTS = 3
NUM_HORIZONTAL_COMPARTMENTS = 2
WAVELENGTHS = np.linspace(700, 900, 41, dtype=int)  # full 41 wavelengths
# WAVELENGTHS = [800]  # one wavelength for testing
NUM_SIMULATIONS = 500

path_manager = sp.PathManager()

# If VISUALIZE is set to True, the simulation result will be plotted
VISUALIZE = False


def create_example_tissue():
    """
    Tissue definition containing muscular background and 0-3 blood vessels.
    """
    dim_x, dim_y, dim_z = settings.get_volume_dimensions_voxels()
    tissue_library = sp.TissueLibrary()

    background_dictionary = sp.Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = tissue_library.constant(1e-10, 1e-10, 1.0)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary

    BLUR_SIZE = 0.2
    tissue_dict["muscle"] = sp.define_horizontal_layer_structure_settings(z_start_mm=SPACING, thickness_mm=100,
                                                                          molecular_composition=
                                                                          tissue_library.muscle(background_oxy=sp.RandomHeterogeneity(dim_x,
                                                                                                                  dim_y, dim_z, SPACING,
                                                                                                                  _gaussian_blur_size_mm=BLUR_SIZE,
                                                                                                                  _min= 1e-5,
                                                                                                                  _max= 1).get_map(),
                                                                              blood_volume_fraction=sp.RandomHeterogeneity(
                                                                                  dim_x, dim_y, dim_z, SPACING, _gaussian_blur_size_mm=BLUR_SIZE,
                                                                                  _min= 1e-2, _max=1).get_map()),
                                                                          priority=1,
                                                                          consider_partial_volume=True,
                                                                          adhere_to_deformation=False)

    return tissue_dict


# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume is generated with the same random seed every time.

for simulation_idx in range(4, NUM_SIMULATIONS):
    # Every volume needs a distinct random seed.
    RANDOM_SEED = int(1e4 + simulation_idx)
    np.random.seed(RANDOM_SEED)
    VOLUME_NAME = "KylieHeterogeneousNoVess2_" + str(RANDOM_SEED)

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
    device.add_illumination_geometry(sp.GaussianBeamIlluminationGeometry(beam_radius_mm=20))

    SIMULATION_PIPELINE = [
        sp.ModelBasedVolumeCreationAdapter(settings),
        sp.MCXAdapter(settings),
    ]

    sp.simulate(SIMULATION_PIPELINE, settings, device)

    if not VISUALIZE:
        pass
    else:
        sp.visualise_data(path_to_hdf5_file=path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5",
                          wavelength=WAVELENGTHS[0],
                          show_initial_pressure=True,
                          show_absorption=True,
                          log_scale=True,
                          show_xz_only=True)
