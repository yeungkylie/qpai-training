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
NUM_VERTICAL_COMPARTMENTS = 1
NUM_HORIZONTAL_COMPARTMENTS = 3
WAVELENGTHS = [800, 805, 810]
NUM_SIMULATIONS = 3

path_manager = sp.PathManager()

# If VISUALIZE is set to True, the simulation result will be plotted
VISUALIZE = True


def create_example_tissue():
    """
    This is a very simple example script of how to create a tissue definition.
    It contains a muscular background, an epidermis layer on top of the muscles
    and a blood vessel.
    """
    background_dictionary = sp.Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-10, 1e-10, 1.0)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary

    # TODO draw two random variables for blood_volume_fraction and blood oxygenation
    # randomise bvf between 0.5-1%
    lower_blood_volume_fraction = 0.005
    upper_blood_volume_fraction = 0.015
    blood_volume_fraction = (lower_blood_volume_fraction - upper_blood_volume_fraction) * np.random.random() + upper_blood_volume_fraction
    # randomise muscle oxygenation to 70% +-10%
    lower_oxy = 0.65
    upper_oxy = 0.75
    oxy = (lower_oxy - upper_oxy) * np.random.random() + upper_oxy
    tissue_dict["muscle"] = sp.define_horizontal_layer_structure_settings(z_start_mm=0, thickness_mm=100,
                                                                          molecular_composition=
                                                                          sp.TISSUE_LIBRARY.muscle(
                                                                              blood_volume_fraction, oxy),
                                                                          priority=1,
                                                                          consider_partial_volume=True,
                                                                          adhere_to_deformation=False)

    for vertical_idx in range(NUM_VERTICAL_COMPARTMENTS):
        for horizontal_idx in range(NUM_HORIZONTAL_COMPARTMENTS):
            idx = vertical_idx * NUM_HORIZONTAL_COMPARTMENTS + horizontal_idx
            # Randomising whether this compartment contains a vessel
            vessel_probability = 0.3
            vessel_randomisation = np.random.random()
            if vessel_randomisation < vessel_probability:
                # TODO: compute the min and max bounds of the x and z dimension based on VOLUME_TRANSDUCER_DIM_IN_MM (x)
                # and VOLUME_HEIGHT_IN_MM (z)
                max_x = VOLUME_TRANSDUCER_DIM_IN_MM * (horizontal_idx + 1) / NUM_HORIZONTAL_COMPARTMENTS
                max_z = VOLUME_HEIGHT_IN_MM * (vertical_idx + 1) / NUM_VERTICAL_COMPARTMENTS
                min_x = max_x - VOLUME_TRANSDUCER_DIM_IN_MM / NUM_HORIZONTAL_COMPARTMENTS
                min_z = max_z - VOLUME_HEIGHT_IN_MM / NUM_VERTICAL_COMPARTMENTS
                # Then create two random variables for the x and z position. The y position is already set correctly
                start_x, end_x = (min_x - max_x) * np.random.random(2) + max_x
                start_z, end_z = (min_z - max_z) * np.random.random(2) + max_z
                # randomise the radius to be somewhere between e.g. 0.3 and 2 mm
                lower_tube_radius = 0.3
                upper_tube_radius = 2
                tube_radius = (lower_tube_radius - upper_tube_radius) * np.random.random() + upper_tube_radius
                # Draw another random variable for the oxygen saturation for each vessel between 0 and 1
                vessel_oxy_sat = np.random.random()
                tissue_dict[f"vessel_{idx}"] = sp.define_circular_tubular_structure_settings(
                    tube_start_mm=[start_x, 0, start_z],
                    tube_end_mm=[end_x, VOLUME_PLANAR_DIM_IN_MM, end_z],
                    molecular_composition=sp.TISSUE_LIBRARY.blood(oxygenation=vessel_oxy_sat),
                    radius_mm=tube_radius, priority=3, consider_partial_volume=True,
                    adhere_to_deformation=False
                )
    return tissue_dict


# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.

for simulation_idx in range(NUM_SIMULATIONS):
    # Every volume needs a distinct random seed.
    RANDOM_SEED = int(1e5 + simulation_idx)

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
        Tags.DO_IPASC_EXPORT: True
    }
    settings = sp.Settings(general_settings)
    np.random.seed(RANDOM_SEED) # this probably needs to be deleted??

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

    settings.set_acoustic_settings({
        Tags.ACOUSTIC_SIMULATION_3D: False,
        Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
        Tags.KWAVE_PROPERTY_ALPHA_POWER: 0.00,
        Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
        Tags.KWAVE_PROPERTY_PMLInside: False,
        Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
        Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
        Tags.KWAVE_PROPERTY_PlotPML: False,
        Tags.RECORDMOVIE: False,
        Tags.MOVIENAME: "visualization_log",
        Tags.ACOUSTIC_LOG_SCALE: True
    })

    settings.set_reconstruction_settings({
        Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: False,
        Tags.TUKEY_WINDOW_ALPHA: 0.5,
        Tags.BANDPASS_CUTOFF_LOWPASS: int(8e6),
        Tags.BANDPASS_CUTOFF_HIGHPASS: int(0.1e4),
        Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION: False,
        Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
        Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_BOX,
        Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_PRESSURE,
        Tags.DATA_FIELD_SPEED_OF_SOUND: 1540,
        Tags.SPACING_MM: SPACING
    })

    device = sp.PhotoacousticDevice(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                                                 VOLUME_PLANAR_DIM_IN_MM / 2,
                                                                 0]),
                                    field_of_view_extent_mm=np.asarray([-25, 25, -10, 10, 0, 20]))
    device.set_detection_geometry(sp.LinearArrayDetectionGeometry(device_position_mm=device.device_position_mm,
                                                                  pitch_mm=0.25,
                                                                  number_detector_elements=76,
                                                                  field_of_view_extent_mm=np.asarray(
                                                                      [-15, 15, -10, 10, 0, 20])))
    device.add_illumination_geometry(sp.GaussianBeamIlluminationGeometry(beam_radius_mm=40))

    SIMULATION_PIPELINE = [
        sp.ModelBasedVolumeCreationAdapter(settings),
        sp.MCXAdapter(settings),
        sp.KWaveAdapter(settings),
        sp.DelayAndSumAdapter(settings),    # image reconstruction
        sp.FieldOfViewCropping(settings)
    ]

    sp.simulate(SIMULATION_PIPELINE, settings, device)

    sp.visualise_data(path_to_hdf5_file=path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5",
                      wavelength=WAVELENGTHS[0],
                      show_initial_pressure=True,
                      show_absorption=True,
                      log_scale=True)