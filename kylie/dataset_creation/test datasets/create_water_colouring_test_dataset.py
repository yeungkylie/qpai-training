from simpa.utils import Tags, TissueLibrary
from simpa.core.simulation import simulate
from simpa.utils.settings import Settings
import simpa as sp
import numpy as np
from simpa import ModelBasedVolumeCreationAdapter, MCXAdapter

from simpa.utils.path_manager import PathManager

# Stop printing out so much.
from simpa.log import Logger
import logging

Logger._logger.setLevel(logging.WARNING)

VOLUME_TRANSDUCER_DIM_IN_MM = 1
VOLUME_PLANAR_DIM_IN_MM = 1
VOLUME_HEIGHT_IN_MM = 15.5
SPACING = 0.05
RANDOM_SEED = 471

# path_manager = PathManager("path_config_water_colouring.env")
path_manager = PathManager()

def create_example_tissue(oxygenation: float = 0.5, depth=0.0):
    """
    Based on the example code. Create a simple skin phantom.
    """
    # Background
    background_dict = Settings()
    water = sp.MOLECULE_LIBRARY.water()
    water.spectrum.values = water.spectrum.values * 10
    water.spectrum.new_absorptions = water.spectrum.new_absorptions * 10
    background_dict[Tags.MOLECULE_COMPOSITION] = sp.MolecularCompositionGenerator().append(
        water).get_molecular_composition(sp.SegmentationClasses.WATER)
    background_dict[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    # Vessel
    vessel_1_dict = Settings()
    vessel_1_dict[Tags.PRIORITY] = 4
    vessel_1_dict[Tags.STRUCTURE_START_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                              0,
                                              depth]
    vessel_1_dict[Tags.STRUCTURE_END_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                            VOLUME_PLANAR_DIM_IN_MM,
                                            depth]
    vessel_1_dict[Tags.STRUCTURE_RADIUS_MM] = 0.35
    vessel_1_dict[Tags.MOLECULE_COMPOSITION] = TissueLibrary().blood(oxygenation=oxygenation)
    vessel_1_dict[Tags.CONSIDER_PARTIAL_VOLUME] = False
    vessel_1_dict[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE
    vessel_1_dict[Tags.ADHERE_TO_DEFORMATION] = False

    tissue_dict = Settings()
    tissue_dict[Tags.BACKGROUND] = background_dict
    tissue_dict["vessel_1"] = vessel_1_dict
    return tissue_dict


DEPTH_LEVELS = [10.0]#[0.5, 2.5, 5.0, 7.5, 10.0, 12.5, 15]
OXYGEN_LEVELS = np.linspace(0, 1, 21)
n_sample = 0

for depth_level in DEPTH_LEVELS:
    for oxygen_fraction in OXYGEN_LEVELS:
        RANDOM_SEED = int(420000 + n_sample)
        np.random.seed(RANDOM_SEED)
        VOLUME_NAME = f"Test_Data_WaterColouring_{n_sample}_oxy_{oxygen_fraction:.2f}_dep_{depth_level:.1f}"
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
            Tags.WAVELENGTHS: [800], #np.linspace(700, 900, 41).astype(np.int64),
            Tags.DIGITAL_DEVICE_POSITION: [VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                           VOLUME_PLANAR_DIM_IN_MM / 2,
                                           0],
            Tags.DO_FILE_COMPRESSION: True
        }
        settings = Settings(general_settings)


        settings.set_volume_creation_settings({
            Tags.SIMULATE_DEFORMED_LAYERS: False,
            Tags.STRUCTURES: create_example_tissue(oxygenation=oxygen_fraction,
                                                   depth=depth_level)
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
        n_sample = n_sample + 1

        sp.visualise_data(settings=settings,
                          path_manager=path_manager,
                          show_initial_pressure=True,
                          show_absorption=True,
                          show_oxygenation=True,
                          show_xz_only=True,
                          log_scale=True)
