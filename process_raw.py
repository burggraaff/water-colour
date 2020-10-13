# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 18:22:30 2020

@author: Burggraaff
"""

"""
Process three images (water, sky, grey card), calibrated using SPECTACLE, to
calculate the remote sensing reflectance in the RGB channels, following the
HydroColor protocol.

Requires the following SPECTACLE calibrations:
    - Metadata
    - Bias
    - Flat-field
    - Spectral response
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, load_camera
from astropy import table
from datetime import datetime, timedelta
from os import walk
from scipy import stats

from wk import hydrocolor as hc, wacodi as wa

# Get the data folder from the command line
calibration_folder, *folders = io.path_from_input(argv)
pattern = calibration_folder.stem

# Get Camera object
camera = load_camera(calibration_folder)
print(f"Loaded Camera object:\n{camera}")

# ISO speed and exposure time are assumed equal between all three images and
# thus can be ignored

# Load effective spectral bandwidths
camera.load_spectral_bands()
effective_bandwidths = camera.spectral_bands[:3]

# Find the effective wavelength corresponding to the RGB bands
RGB_wavelengths = hc.effective_wavelength(calibration_folder)

for folder_main in folders:
    for tup in walk(folder_main):
        folder = io.Path(tup[0])
        data_path = folder/pattern
        if not data_path.exists():
            continue

        # Load data
        water_path, sky_path, card_path = hc.generate_paths(data_path, camera.raw_extension)
        water_raw, sky_raw, card_raw = hc.load_raw_images(water_path, sky_path, card_path)
        print("Loaded RAW data")

        # Load EXIF data
        water_exif, sky_exif, card_exif = hc.load_exif(water_path, sky_path, card_path)

        # Load thumbnails
        water_jpeg, sky_jpeg, card_jpeg = hc.load_raw_thumbnails(water_path, sky_path, card_path)
        print("Created JPEG thumbnails")

        # Correct for bias
        water_bias, sky_bias, card_bias = camera.correct_bias(water_raw, sky_raw, card_raw)
        print("Corrected bias")

        # Normalising for ISO speed is not necessary since this is a relative measurement

        # Dark current is negligible

        # Correct for flat-field
        water_flat, sky_flat, card_flat = camera.correct_flatfield(water_bias, sky_bias, card_bias)
        print("Corrected flat-field")

        # Demosaick the data
        water_RGBG, sky_RGBG, card_RGBG = camera.demosaick(water_flat, sky_flat, card_flat)
        print("Demosaicked")

        # Select the central pixels
        water_cut, sky_cut, card_cut = hc.central_slice_raw(water_RGBG, sky_RGBG, card_RGBG)

        # Combined histograms of different data reduction steps
        water_all = [water_jpeg, water_raw, water_bias, water_flat, water_cut]
        sky_all = [sky_jpeg, sky_raw, sky_bias, sky_flat, sky_cut]
        card_all = [card_jpeg, card_raw, card_bias, card_flat, card_cut]

        hc.histogram_raw(water_all, sky_all, card_all, camera=camera, saveto=data_path/"statistics_raw.pdf")

        # Flatten lists and combine G and G2
        water_RGB, sky_RGB, card_RGB = hc.RGBG2_to_RGB(water_cut, sky_cut, card_cut)

        water_mean = np.array([rgb.mean() for rgb in water_RGB])
        sky_mean = np.array([rgb.mean() for rgb in sky_RGB])
        card_mean = np.array([rgb.mean() for rgb in card_RGB])
        print("Calculated mean values per channel")

        water_std = np.array([rgb.std() for rgb in water_RGB])
        sky_std = np.array([rgb.std() for rgb in sky_RGB])
        card_std = np.array([rgb.std() for rgb in card_RGB])
        print("Calculated standard deviations per channel")

        water_err = np.array([stats.sem(rgb) for rgb in water_RGB])
        sky_err = np.array([stats.sem(rgb) for rgb in sky_RGB])
        card_err = np.array([stats.sem(rgb) for rgb in card_RGB])
        print("Calculated standard errors per channel")

        # HydroColor

        # Convert to remote sensing reflectances
        R_rs = hc.R_RS(water_mean, sky_mean, card_mean)
        print("Calculated remote sensing reflectances")

        R_rs_err = hc.R_RS_error(water_mean, sky_mean, card_mean, water_std, sky_std, card_std)
        print("Calculated error in remote sensing reflectances")

        for R, R_err, c in zip(R_rs, R_rs_err, "RGB"):
            print(f"{c}: R_rs = {R:.3f} +- {R_err:.3f} sr^-1")

        # Plot the result
        hc.plot_R_rs(RGB_wavelengths, R_rs, effective_bandwidths, R_rs_err)

        # WACODI

        # Convert RGB to XYZ
        water_XYZ, sky_XYZ, card_XYZ = camera.convert_to_XYZ(water_mean, sky_mean, card_mean)
        water_XYZ_err, sky_XYZ_err, card_XYZ_err = wa.convert_errors_to_XYZ(camera.XYZ_matrix, water_std, sky_std, card_std)

        # Calculate xy chromaticity
        water_xy, sky_xy, card_xy = wa.convert_XYZ_to_xy(water_XYZ, sky_XYZ, card_XYZ)

        # Calculate hue angle
        water_hue, sky_hue, card_hue = wa.convert_xy_to_hue_angle(water_xy, sky_xy, card_xy)
        water_hue_err, sky_hue_err, card_hue_err = [wa.convert_XYZ_error_to_hue_angle(XYZ_data, XYZ_error) for XYZ_data, XYZ_error in zip([water_XYZ, sky_XYZ, card_XYZ], [water_XYZ_err, sky_XYZ_err, card_XYZ_err])]

        # Create a timestamp from EXIF (assume time zone UTC+2)
        UTC = hc.UTC_timestamp(water_exif)

        # Write the result to file
        hc.write_R_rs(UTC, R_rs, R_rs_err, saveto=data_path.parent / (data_path.stem + "_raw.csv"))
