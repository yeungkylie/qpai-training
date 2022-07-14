from simpa import Tags
import simpa as sp
import numpy as np

# FIXME temporary workaround for newest Intel architectures
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

VOLUME_TRANSDUCER_DIM_IN_MM = 19.2
VOLUME_PLANAR_DIM_IN_MM = 19.2
VOLUME_HEIGHT_IN_MM = 19.2
SPACING = 0.3
RANDOM_SEED = 0
WAVELENGTHS = np.linspace(700, 900, 41, dtype=int)  # full 41 wavelengths
path_manager = sp.PathManager()

VISUALIZE = True

np.random.seed(RANDOM_SEED)
VOLUME_NAME = "KylieBaseline_"+str(1e4+RANDOM_SEED)

general_settings = {
            # These parameters set the general properties of the simulated volume
            Tags.RANDOM_SEED: RANDOM_SEED,
            Tags.VOLUME_NAME: "KylieBaseline_" + str(1e4+RANDOM_SEED),
            Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
            Tags.SPACING_MM: SPACING,
            Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
            Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
            Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
            Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,
            Tags.GPU: True,
            Tags.WAVELENGTHS: [700, 800],
            Tags.DO_FILE_COMPRESSION: True,
            Tags.DO_IPASC_EXPORT: True
        }
settings = sp.Settings(general_settings)
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
    Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
    Tags.ACOUSTIC_SIMULATION_3D: False,
    Tags.KWAVE_PROPERTY_ALPHA_POWER: 0.00,
    Tags.TUKEY_WINDOW_ALPHA: 0.5,
    Tags.BANDPASS_CUTOFF_LOWPASS: int(8e6),
    Tags.BANDPASS_CUTOFF_HIGHPASS: int(0.1e4),
    Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION: False,
    Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
    Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_BOX,
    Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_PRESSURE,
    Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
    Tags.KWAVE_PROPERTY_PMLInside: False,
    Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
    Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
    Tags.KWAVE_PROPERTY_PlotPML: False,
    Tags.RECORDMOVIE: False,
    Tags.MOVIENAME: "visualization_log",
    Tags.ACOUSTIC_LOG_SCALE: True,
    Tags.DATA_FIELD_SPEED_OF_SOUND: 1540,
    Tags.DATA_FIELD_ALPHA_COEFF: 0.01,
    Tags.DATA_FIELD_DENSITY: 1000,
    Tags.SPACING_MM: SPACING
})
device = sp.PhotoacousticDevice(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                                             VOLUME_PLANAR_DIM_IN_MM / 2,
                                                             0]),
                                field_of_view_extent_mm=np.asarray([-15, 15, 0, 0, 0, 20]))
device.set_detection_geometry(sp.LinearArrayDetectionGeometry(device_position_mm=device.device_position_mm,
                                                              pitch_mm=0.25,
                                                              number_detector_elements=50,
                                                              field_of_view_extent_mm=np.asarray([-15, 15, 0, 0, 0, 20])))
print(device.get_detection_geometry().get_detector_element_positions_base_mm())
device.add_illumination_geometry(sp.GaussianBeamIlluminationGeometry(beam_radius_mm=20))

SIMULATION_PIPELINE = [
    sp.KWaveAdapter(settings),
    sp.TimeReversalAdapter(settings),
    ]

sp.simulate(SIMULATION_PIPELINE, settings, device)

if Tags.WAVELENGTH in settings:
    WAVELENGTH = settings[Tags.WAVELENGTH]
else:
    WAVELENGTH = 700

if VISUALIZE:
    sp.visualise_data(path_to_hdf5_file=settings[Tags.SIMPA_OUTPUT_PATH],
                      wavelength=WAVELENGTH,
                      show_time_series_data=True,
                      show_initial_pressure=True,
                      show_reconstructed_data=True,
                      log_scale=False,
                      show_xz_only=False)
