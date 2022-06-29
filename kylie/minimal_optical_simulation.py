# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa import Tags
import simpa as sp
import numpy as np

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = sp.PathManager()

VOLUME_TRANSDUCER_DIM_IN_MM = 19.2
VOLUME_PLANAR_DIM_IN_MM = 19.2
VOLUME_HEIGHT_IN_MM = 19.2
SPACING = 0.3
RANDOM_SEED = 471
VOLUME_NAME = "KylieBaselineTest_"+str(RANDOM_SEED)
SAVE_REFLECTANCE = False
SAVE_PHOTON_DIRECTION = False

# If VISUALIZE is set to True, the simulation result will be plotted
VISUALIZE = True


def create_example_tissue():
    """
    This is a very simple example script of how to create a tissue definition.
    It contains a muscular background, an epidermis layer on top of the muscles
    and a blood vessel.
    """
    background_dictionary = sp.Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-4, 1e-4, 0.9)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    # randomise bvf between 0.5-1%
    lower_blood_volume_fraction = 0.005
    upper_blood_volume_fraction = 0.01
    blood_volume_fraction = (lower_blood_volume_fraction - upper_blood_volume_fraction) * np.random.random() + upper_blood_volume_fraction
    # randomise muscle oxygenation to 70% +-10%
    lower_oxy = 0.65
    upper_oxy = 0.75
    oxy = (lower_oxy - upper_oxy) * np.random.random() + upper_oxy
    muscle_dictionary = sp.Settings()
    muscle_dictionary[Tags.PRIORITY] = 1
    muscle_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 0]
    muscle_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 100]
    muscle_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.muscle(oxy, blood_volume_fraction)
    muscle_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    muscle_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
    muscle_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    vessel_1_dictionary = sp.Settings()
    vessel_1_dictionary[Tags.PRIORITY] = 3
    vessel_1_dictionary[Tags.STRUCTURE_START_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                    10,
                                                    VOLUME_HEIGHT_IN_MM/2]
    vessel_1_dictionary[Tags.STRUCTURE_END_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                  12,
                                                  VOLUME_HEIGHT_IN_MM/2]
    vessel_1_dictionary[Tags.STRUCTURE_RADIUS_MM] = 3
    vessel_1_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.blood()
    vessel_1_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    vessel_1_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    # epidermis_dictionary = sp.Settings()
    # epidermis_dictionary[Tags.PRIORITY] = 8
    # epidermis_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 9]
    # epidermis_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 10]
    # epidermis_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.epidermis()
    # epidermis_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    # epidermis_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
    # epidermis_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary
    tissue_dict["muscle"] = muscle_dictionary
    # tissue_dict["epidermis"] = epidermis_dictionary
    tissue_dict["vessel_1"] = vessel_1_dictionary
    return tissue_dict


# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.

np.random.seed(RANDOM_SEED)

general_settings = {
    # These parameters set the general properties of the simulated volume
    Tags.RANDOM_SEED: RANDOM_SEED,
    Tags.VOLUME_NAME: VOLUME_NAME,
    Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
    Tags.SPACING_MM: SPACING,
    Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
    Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
    Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
    Tags.WAVELENGTHS: [798],
    Tags.DO_FILE_COMPRESSION: True
}

settings = sp.Settings(general_settings)

settings.set_volume_creation_settings({
    Tags.SIMULATE_DEFORMED_LAYERS: True,
    Tags.STRUCTURES: create_example_tissue()
})
settings.set_optical_settings({
    Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
    Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
    Tags.COMPUTE_DIFFUSE_REFLECTANCE: SAVE_REFLECTANCE,
    Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT: SAVE_PHOTON_DIRECTION
})
settings["noise_model_1"] = {
    Tags.NOISE_MEAN: 1.0,
    Tags.NOISE_STD: 0.1,
    Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
    Tags.DATA_FIELD: Tags.DATA_FIELD_INITIAL_PRESSURE,
    Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
}

if not SAVE_REFLECTANCE and not SAVE_PHOTON_DIRECTION:
    pipeline = [
        sp.ModelBasedVolumeCreationAdapter(settings),
        sp.MCXAdapter(settings),
        sp.GaussianNoise(settings, "noise_model_1")
    ]
else:
    pipeline = [
        sp.ModelBasedVolumeCreationAdapter(settings),
        sp.MCXAdapterReflectance(settings),
    ]


# class ExampleDeviceSlitIlluminationLinearDetector(sp.PhotoacousticDevice):
#     """
#     This class represents a digital twin of a PA device with a slit as illumination next to a linear detection geometry.
#
#     """
#
#     def __init__(self):
#         super().__init__(device_position_mm=np.asarray([VOLUME_TRANSDUCER_DIM_IN_MM/2,
#                                                         VOLUME_PLANAR_DIM_IN_MM/2, 0]))
#         self.set_detection_geometry(sp.LinearArrayDetectionGeometry())
#         self.add_illumination_geometry(sp.SlitIlluminationGeometry(slit_vector_mm=[20, 0, 0],
#                                                                    direction_vector_mm=[0, 0, 1]))
#
#
# device = ExampleDeviceSlitIlluminationLinearDetector()

device = sp.PhotoacousticDevice(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                                             VOLUME_PLANAR_DIM_IN_MM / 2,
                                                             0]),
                                field_of_view_extent_mm=np.asarray([-25, 25, -10, 10, 0, 20]))
# device.set_detection_geometry(sp.LinearArrayDetectionGeometry(device_position_mm=device.device_position_mm,
#                                                               pitch_mm=0.25,
#                                                               number_detector_elements=76,
#                                                               field_of_view_extent_mm=np.asarray(
#                                                                   [-15, 15, -10, 10, 0, 20])))
device.add_illumination_geometry(sp.GaussianBeamIlluminationGeometry(beam_radius_mm=40))

sp.simulate(pipeline, settings, device)

if Tags.WAVELENGTH in settings:
    WAVELENGTH = settings[Tags.WAVELENGTH]
else:
    WAVELENGTH = 700

if VISUALIZE:
    sp.visualise_data(path_to_hdf5_file=path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5",
                      wavelength=WAVELENGTH,
                      show_initial_pressure=True,
                      show_absorption=True,
                      show_diffuse_reflectance=SAVE_REFLECTANCE,
                      log_scale=True)
