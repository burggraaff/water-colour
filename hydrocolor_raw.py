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
from spectacle import io, calibrate, analyse, spectral
from astropy import table
from datetime import datetime
from os import walk

# Get the data folder from the command line
calibration_folder, *folders = io.path_from_input(argv)

pattern = calibration_folder.stem

# Get metadata
camera = io.load_metadata(calibration_folder)
print("Loaded metadata")

# ISO speed and exposure time are assumed equal between all three images and
# thus can be ignored

for folder_main in folders:
    for tup in walk(folder_main):
        folder = io.Path(tup[0])
        data_path = folder/pattern
        if not data_path.exists():
            continue

        # Load data
        water_path, sky_path, card_path = [data_path/(photo + camera.image.raw_extension) for photo in ("water", "sky", "greycard")]
        water_raw = io.load_raw_image(water_path)
        sky_raw = io.load_raw_image(sky_path)
        card_raw = io.load_raw_image(card_path)
        print("Loaded RAW data")

        # Load EXIF data
        water_exif = io.load_exif(water_path)
        sky_exif = io.load_exif(sky_path)
        card_exif = io.load_exif(card_path)

        # Load thumbnails
        water_jpeg = io.load_jpg_image(water_path)
        sky_jpeg = io.load_jpg_image(sky_path)
        card_jpeg = io.load_jpg_image(card_path)
        print("Loaded JPEG thumbnails")

        # Correct for bias
        water_bias, sky_bias, card_bias = calibrate.correct_bias(calibration_folder, water_raw, sky_raw, card_raw)
        print("Corrected bias")

        # Normalising for ISO speed is not necessary since this is a relative measurement

        # Dark current is negligible

        # Correct for flat-field
        water_flat, sky_flat, card_flat = calibrate.correct_flatfield(calibration_folder, water_bias, sky_bias, card_bias)
        print("Corrected flat-field")

        # Demosaick the data
        water_RGBG, sky_RGBG, card_RGBG = camera.demosaick(water_flat, sky_flat, card_flat)
        print("Demosaicked")

        # Select the central 100x100 pixels
        central_x, central_y = water_RGBG.shape[1]//2, water_RGBG.shape[2]//2
        box_size = 100
        half_box = box_size // 2
        central_slice = np.s_[:, central_x-half_box:central_x+half_box+1, central_y-half_box:central_y+half_box+1]
        water_cut = water_RGBG[central_slice]
        sky_cut = sky_RGBG[central_slice]
        card_cut = card_RGBG[central_slice]
        print(f"Selected central {box_size}x{box_size} pixels")

        # Combined histograms of different data reduction steps
        water_all = [water_raw, water_bias, water_flat, water_cut]
        sky_all = [sky_raw, sky_bias, sky_flat, sky_cut]
        card_all = [card_raw, card_bias, card_flat, card_cut]

        fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(11,4), tight_layout=True, gridspec_kw={"hspace": 0, "wspace": 0}, sharex="col", sharey="col")

        for ax_col, water, sky, card in zip(axs[:,1:].T, water_all, sky_all, card_all):
            data_combined = np.ravel([water, sky, card])
            xmin, xmax = analyse.symmetric_percentiles(data_combined, percent=0.001)
            bins = np.linspace(xmin, xmax, 150)

            for ax, data in zip(ax_col, [water, sky, card]):
                ax.hist(data.ravel(), bins=bins, color="k")
                ax.set_xlim(xmin, xmax)
                ax.grid(True, ls="--", alpha=0.7)

        for ax, img in zip(axs[:,0], [water_jpeg, sky_jpeg, card_jpeg]):
            ax.imshow(img)
            ax.tick_params(bottom=False, labelbottom=False)
        for ax in axs.ravel():
            ax.tick_params(left=False, labelleft=False)
        for ax, label in zip(axs[:,0], ["Water", "Sky", "Grey card"]):
            ax.set_ylabel(label)
        for ax, title in zip(axs[0], ["Image", "Raw", "Bias-corrected", "Flat-fielded", "Central slice"]):
            ax.set_title(title)

        plt.savefig(data_path/"statistics_raw.pdf")
        plt.show()

        # Convert to remote sensing reflectances
        def R_RS(L_t, L_s, L_c, rho=0.028, R_ref=0.18):
            return (L_t - rho * L_s) / ((np.pi / R_ref) * L_c)

        def RGBG2_to_RGB(array):
            return [array[0].ravel(), array[1::2].ravel(), array[2].ravel()]

        # Flatten lists and combine G and G2
        water_RGB = RGBG2_to_RGB(water_cut)
        sky_RGB = RGBG2_to_RGB(sky_cut)
        card_RGB = RGBG2_to_RGB(card_cut)

        water_mean = np.array([rgb.mean() for rgb in water_RGB])
        sky_mean = np.array([rgb.mean() for rgb in sky_RGB])
        card_mean = np.array([rgb.mean() for rgb in card_RGB])
        print("Calculated mean values per channel")

        water_std = np.array([rgb.std() for rgb in water_RGB])
        sky_std = np.array([rgb.std() for rgb in sky_RGB])
        card_std = np.array([rgb.std() for rgb in card_RGB])

        R_rs = R_RS(water_mean, sky_mean, card_mean)
        print("Calculated remote sensing reflectances")


        R_rs_err_water = water_std**2 * ((0.18/np.pi) * card_mean**-1)**2
        R_rs_err_sky = sky_std**2 * ((0.18/np.pi) * 0.028 * card_mean**-1)**2
        R_rs_err_card = card_std**2 * ((0.18/np.pi) * (water_mean - 0.028 * sky_mean) * card_mean**-2)**2

        R_rs_err = np.sqrt(R_rs_err_water + R_rs_err_sky + R_rs_err_card)
        print("Calculated error in remote sensing reflectances")

        for R, R_err, c in zip(R_rs, R_rs_err, "RGB"):
            print(f"{c}: R_rs = {R:.3f} +- {R_err:.3f} sr^-1")

        # Find the effective wavelength corresponding to the RGB bands
        spectral_response = calibrate.load_spectral_response(calibration_folder)
        wavelengths = spectral_response[0]
        RGB_responses = spectral_response[1:4]
        RGB_wavelengths = spectral.effective_wavelengths(wavelengths, RGB_responses)

        # SPECTACLE function for effective bandwidth is currently somewhat broken so we
        # do it ourselves
        RGB_responses_normalised = RGB_responses / RGB_responses.max(axis=1)[:,np.newaxis]
        effective_bandwidths = np.trapz(RGB_responses_normalised, x=wavelengths, axis=1)

        plt.figure(figsize=(3,3), tight_layout=True)
        for j, c in enumerate("rgb"):
            plt.errorbar(RGB_wavelengths[j], R_rs[j], xerr=effective_bandwidths[j]/2, yerr=R_rs_err[j], c=c, fmt="o")
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Remote sensing reflectance [sr$^{-1}$]")
        plt.xlim(390, 700)
        plt.ylim(0, 0.05)
        plt.yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
        plt.grid(ls="--")
        plt.show()

        # Create a timestamp
        time = data_path.parents[0].stem[3:]
        hour, minute = time[:2], time[3:]
        date = data_path.parents[2].stem
        year, month, day = date[:4], date[4:6], date[6:]
        year, month, day, hour, minute = [int(x) for x in (year, month, day, hour, minute)]
        timestamp = datetime(year, month, day, hour, minute, second=0)
        timestamp_iso = timestamp.isoformat()

        # Write the result to file
        result = table.Table(rows=[[timestamp_iso, *R_rs, *R_rs_err]], names=["UTC", "R_rs (R)", "R_rs (G)", "R_rs (B)", "R_rs_err (R)", "R_rs_err (G)", "R_rs_err (B)"])
        save_to = data_path.parent / (data_path.stem + "_raw.csv")
        result.write(save_to, format="ascii.fast_csv")

        print(f"Saved results to {save_to}")
