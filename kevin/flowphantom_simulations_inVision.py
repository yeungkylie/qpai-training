from simpa.utils import Tags, TISSUE_LIBRARY
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

# FIXME temporary workaround for newest Intel architectures
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

VOLUME_TRANSDUCER_DIM_IN_MM = 90
VOLUME_PLANAR_DIM_IN_MM = 40
VOLUME_HEIGHT_IN_MM = 90
SPACING = 0.5
RANDOM_SEED = 471

path_manager = sp.PathManager()


def create_example_tissue(melanosome: float = 0., oxygenation: float = 0.9):
    """
    Based on the example code. Create a simple skin phantom.
    """
    # Background
    background_dictionary = Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.MolecularCompositionGenerator().append(
        sp.MOLECULE_LIBRARY.water()).get_molecular_composition(sp.SegmentationClasses.WATER)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    intralipid = sp.Molecule(volume_fraction=1.0,
                             anisotropy_spectrum=sp.AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(0.9),
                             scattering_spectrum=sp.ScatteringSpectrumLibrary.scattering_from_rayleigh_and_mie_theory(
                                 name="Intralipid", mus_at_500_nm=117,
                                 mie_power_law_coefficient=2.33, fraction_rayleigh_scattering=0
                             ),
                             absorption_spectrum=sp.MOLECULE_LIBRARY.water().spectrum,
                             speed_of_sound=sp.utils.StandardProperties.SPEED_OF_SOUND_WATER,
                             gruneisen_parameter=1.0,
                             name="Intralipid")

    # Muscle
    muscle_dictionary = Settings()
    muscle_dictionary[Tags.PRIORITY] = 3
    muscle_dictionary[Tags.STRUCTURE_START_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM / 2, 0, VOLUME_HEIGHT_IN_MM / 2]
    muscle_dictionary[Tags.STRUCTURE_END_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM / 2, 40, VOLUME_HEIGHT_IN_MM / 2]
    muscle_dictionary[Tags.MOLECULE_COMPOSITION] = sp.MolecularCompositionGenerator().append(
                            intralipid).get_molecular_composition(sp.SegmentationClasses.MUSCLE)
    muscle_dictionary[Tags.STRUCTURE_RADIUS_MM] = 9
    muscle_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    muscle_dictionary[Tags.ADHERE_TO_DEFORMATION] = False
    muscle_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    # Vessel
    vessel_1_dictionary = Settings()
    vessel_1_dictionary[Tags.PRIORITY] = 4
    vessel_1_dictionary[Tags.STRUCTURE_START_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                                    0,
                                                    VOLUME_HEIGHT_IN_MM / 2]
    vessel_1_dictionary[Tags.STRUCTURE_END_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                                  40,
                                                  VOLUME_HEIGHT_IN_MM / 2]
    vessel_1_dictionary[Tags.STRUCTURE_RADIUS_MM] = 0.75
    vessel_1_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood(oxygenation=oxygenation)
    vessel_1_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = False
    vessel_1_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE
    vessel_1_dictionary[Tags.ADHERE_TO_DEFORMATION] = False

    # Epidermis
    epidermis_dictionary = Settings()
    epidermis_dictionary[Tags.PRIORITY] = 2
    epidermis_dictionary[Tags.STRUCTURE_START_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM / 2, 0, VOLUME_HEIGHT_IN_MM / 2]
    epidermis_dictionary[Tags.STRUCTURE_END_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM / 2, 40, VOLUME_HEIGHT_IN_MM / 2]
    epidermis_dictionary[Tags.STRUCTURE_RADIUS_MM] = 10
    epidermis_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.epidermis(melanosom_volume_fraction=melanosome)
    epidermis_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    epidermis_dictionary[Tags.ADHERE_TO_DEFORMATION] = False
    epidermis_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    tissue_dict = Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary
    tissue_dict["muscle"] = muscle_dictionary
    #tissue_dict["epidermis"] = epidermis_dictionary
    tissue_dict["vessel_1"] = vessel_1_dictionary
    return tissue_dict


# MELANIN_LEVELS = [0.00, 0.005, 0.01, 0.015, 0.02]
# OXYGEN_LEVELS = np.linspace(0, 1, 11)
NUM_SAMPLES = 3
# for n_sample in range(505,NUM_SAMPLES)
n_sample = 505
print("Computing run:", n_sample)
VOLUME_NAME = f"Flowphantom_inVision_700_900_5_Water_{n_sample:4.0f}"
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
    Tags.WAVELENGTHS: np.linspace(700, 900, 6).astype(np.int64),
    Tags.DIGITAL_DEVICE_POSITION: [VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                   VOLUME_PLANAR_DIM_IN_MM / 2,
                                   0],
    Tags.DO_FILE_COMPRESSION: True
}
settings = Settings(general_settings)

RANDOM_SEED = int(170000 + n_sample)
np.random.seed(RANDOM_SEED)

oxy = np.random.random()
print("OXY", oxy)

settings.set_volume_creation_settings({
    Tags.SIMULATE_DEFORMED_LAYERS: False,
    Tags.STRUCTURES: create_example_tissue(oxygenation=oxy)
})
settings.set_optical_settings({
    Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
    Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
    Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_MCX,
    Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_PENCIL,
    Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
    Tags.MCX_ASSUMED_ANISOTROPY: 0.9
})

pipeline = [
    ModelBasedVolumeCreationAdapter(settings),
    MCXAdapter(settings)
]
device = sp.InVision256TF(device_position_mm=np.asarray([VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                                         VOLUME_PLANAR_DIM_IN_MM / 2,
                                                         VOLUME_HEIGHT_IN_MM / 2]),
                          field_of_view_extent_mm=np.asarray([-12, 12, 0, 0, -12, 12]))

simulate(pipeline, settings, device)
