"""
Process the images, calibrated with SPECTACLE, through the WACODI method.

Requires the following SPECTACLE calibrations:
    - Metadata
    - Bias
    - Flat-field
    - RGB to XYZ matrix
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, load_camera, spectral
from astropy import table
from datetime import datetime, timedelta
from os import walk

from wk import hydrocolor as hc

# Get the data folder from the command line
calibration_folder, *folders = io.path_from_input(argv)
pattern = calibration_folder.stem

# Get Camera object
camera = load_camera(calibration_folder)
print(f"Loaded Camera object:\n{camera}")

# ISO speed and exposure time are assumed equal between all three images and
# thus can be ignored

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

        # hc.histogram_raw(water_all, sky_all, card_all, camera=camera, saveto=data_path/"statistics_raw.pdf")

        water_RGB, sky_RGB, card_RGB = hc.RGBG2_to_RGB(water_cut, sky_cut, card_cut)

        water_mean = np.array([rgb.mean() for rgb in water_RGB])
        sky_mean = np.array([rgb.mean() for rgb in sky_RGB])
        card_mean = np.array([rgb.mean() for rgb in card_RGB])
        print("Calculated mean values per channel")

        water_std = np.array([rgb.std() for rgb in water_RGB])
        sky_std = np.array([rgb.std() for rgb in sky_RGB])
        card_std = np.array([rgb.std() for rgb in card_RGB])
        print("Calculated standard deviations per channel")

        # Convert RGB to XYZ
        water_XYZ, sky_XYZ, card_XYZ = camera.convert_to_XYZ(water_mean, sky_mean, card_mean)

        # Calculate xy chromaticity
        water_xy = (water_XYZ / water_XYZ.sum(axis=0))[:2]
        sky_xy = (sky_XYZ / sky_XYZ.sum(axis=0))[:2]
        card_xy = (card_XYZ / card_XYZ.sum(axis=0))[:2]

        # Calculate hue angle
        water_hue = np.rad2deg(np.arctan2(water_xy[1]-1/3, water_xy[0]-1/3) % (2*np.pi))
        sky_hue = np.rad2deg(np.arctan2(sky_xy[1]-1/3, sky_xy[0]-1/3) % (2*np.pi))
        card_hue = np.rad2deg(np.arctan2(card_xy[1]-1/3, card_xy[0]-1/3) % (2*np.pi))
