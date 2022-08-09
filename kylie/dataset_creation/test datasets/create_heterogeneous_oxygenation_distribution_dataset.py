from simpa.utils import Tags, TissueLibrary
from simpa.core.simulation import simulate
from simpa.utils.settings import Settings
from simpa.core.device_digital_twins import *
import simpa as sp
import numpy as np
from simpa import ModelBasedVolumeCreationAdapter, MCXAdapter

from simpa.utils.path_manager import PathManager

# Stop printing out so much.
from simpa.log import Logger
import logging

Logger._logger.setLevel(logging.WARNING)

VOLUME_TRANSDUCER_DIM_IN_MM = 2
VOLUME_PLANAR_DIM_IN_MM = 2
VOLUME_HEIGHT_IN_MM = 2
SPACING = 0.05
RANDOM_SEED = 471

# path_manager = PathManager("path_config_heterogeneous.env")
path_manager = sp.PathManager()

def create_example_tissue(settings: sp.Settings):
    """
    Based on the example code. Create a simple skin phantom.
    """
    # Background
    xdim, ydim, zdim = settings.get_volume_dimensions_voxels()
    background_dict = Settings()
    oxy_map = sp.RandomHeterogeneity(_spacing_mm=SPACING,
                                     _gaussian_blur_size_mm=0.2,
                                     _min=0.0,
                                     _max=1.0,
                                     _xdim=xdim,
                                     _ydim=ydim,
                                     _zdim=zdim).get_map()
    fraction_map = sp.RandomHeterogeneity(_spacing_mm=SPACING,
                                          _gaussian_blur_size_mm=0.2,
                                          _min=0.01,
                                          _max=0.5,
                                          _xdim=xdim,
                                          _ydim=ydim,
                                          _zdim=zdim).get_map()

    print(np.mean(oxy_map), np.std(oxy_map), np.min(oxy_map), np.max(oxy_map))
    print(np.mean(fraction_map), np.std(fraction_map), np.min(fraction_map), np.max(fraction_map))
    background_dict[Tags.MOLECULE_COMPOSITION] = TissueLibrary().muscle(background_oxy=oxy_map,
                                                                        blood_volume_fraction=fraction_map)
    background_dict[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    # Create the tissue dictionary
    tissue_dict = Settings()
    tissue_dict[Tags.BACKGROUND] = background_dict
    return tissue_dict

# for n_sample in range(250):
for n_sample in range(1):
    RANDOM_SEED = int(230000 + n_sample)
    np.random.seed(RANDOM_SEED)
    VOLUME_NAME = f"Test_Data_Heterogeneous_{n_sample}"
    VISUALIZE = True
    general_settings = {
        # These parameters set the general properties of the simulated volume
        Tags.RANDOM_SEED: RANDOM_SEED,
        Tags.VOLUME_NAME: VOLUME_NAME,
        Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
        Tags.SPACING_MM: SPACING,
        Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
        Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
        Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
        # Tags.WAVELENGTHS: np.linspace(700, 900, 41).astype(np.int64),
        Tags.WAVELENGTHS: [700],
        Tags.DIGITAL_DEVICE_POSITION: [VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                       VOLUME_PLANAR_DIM_IN_MM / 2,
                                       0],
        Tags.DO_FILE_COMPRESSION: True
    }
    settings = Settings(general_settings)

    settings.set_volume_creation_settings({
        Tags.SIMULATE_DEFORMED_LAYERS: False,
        Tags.STRUCTURES: create_example_tissue(settings)
    })
    settings.set_optical_settings({
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
        Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
        Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_MCX,
        Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
        Tags.MCX_ASSUMED_ANISOTROPY: 0.9
    })

    pipeline = [
        ModelBasedVolumeCreationAdapter(settings),
        MCXAdapter(settings)
    ]
    device = sp.PhotoacousticDevice(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                                                 VOLUME_PLANAR_DIM_IN_MM / 2,
                                                                 0]),
                                    field_of_view_extent_mm=np.asarray([-1, 1, 0, 0, 0, 2]))
    device.add_illumination_geometry(sp.GaussianBeamIlluminationGeometry(beam_radius_mm=1))

    simulate(pipeline, settings, device)

    sp.visualise_data(settings=settings,
                      path_manager=path_manager,
                      show_initial_pressure=True,
                      show_absorption=True,
                      show_oxygenation=True,
                      show_xz_only=True,
                      log_scale=True)
