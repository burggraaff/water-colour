"""
Process three images (water, sky, grey card), using JPEG data, to
calculate the remote sensing reflectance in the RGB channels, following the
HydroColor protocol.
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, calibrate, spectral
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
        water_path, sky_path, card_path = [data_path/(photo + ".JPG") for photo in ("water", "sky", "greycard")]
        water_jpeg = io.load_jpg_image(water_path)
        sky_jpeg = io.load_jpg_image(sky_path)
        card_jpeg = io.load_jpg_image(card_path)
        print("Loaded JPEG data")

        # Select the central 100x100 pixels
        central_x, central_y = water_jpeg.shape[0]//2, water_jpeg.shape[1]//2
        box_size = 100
        central_slice = np.s_[central_x-box_size:central_x+box_size+1, central_y-box_size:central_y+box_size+1]
        water_cut = water_jpeg[central_slice]
        sky_cut = sky_jpeg[central_slice]
        card_cut = card_jpeg[central_slice]
        print(f"Selected central {2*box_size}x{2*box_size} pixels")

        # Select the central pixels in the JPEG images
        central_x, central_y = water_jpeg.shape[0]//2, water_jpeg.shape[1]//2
        central_slice = np.s_[central_x-box_size:central_x+box_size+1, central_y-box_size:central_y+box_size+1, :]
        water_jpeg_cut = water_jpeg[central_slice]
        sky_jpeg_cut = sky_jpeg[central_slice]
        card_jpeg_cut = card_jpeg[central_slice]
        print(f"Selected central {2*box_size}x{2*box_size} pixels in the JPEG data")

        # Combined histograms of different data reduction steps
        water_all = [water_jpeg, water_cut]
        sky_all = [sky_jpeg, sky_cut]
        card_all = [card_jpeg, card_cut]

        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9,4), tight_layout=True, gridspec_kw={"hspace": 0, "wspace": 0}, sharex="col", sharey="col")

        for ax_col, water, sky, card in zip(axs.T[1:], water_all, sky_all, card_all):
            data_combined = np.ravel([water, sky, card])
            bins = np.linspace(0, 255, 100)

            for ax, data in zip(ax_col, [water, sky, card]):
                ax.hist(data.ravel(), bins=bins, color="k")
                ax.set_xlim(0, 255)
                ax.grid(True, ls="--", alpha=0.7)

        for ax, image in zip(axs[:,0], [water_jpeg, sky_jpeg, card_jpeg]):
            ax.imshow(image)
            ax.tick_params(bottom=False, labelbottom=False)

        for ax in axs.ravel():
            ax.tick_params(left=False, labelleft=False)
        for ax, label in zip(axs[:,0], ["Water", "Sky", "Grey card"]):
            ax.set_ylabel(label)
        for ax, title in zip(axs[0], ["Image", "JPEG (full)", "Central slice"]):
            ax.set_title(title)

        plt.savefig(data_path/"statistics_jpeg.pdf")
        plt.show()

        # Convert to remote sensing reflectances
        def R_RS(L_t, L_s, L_c, rho=0.028, R_ref=0.18):
            return (L_t - rho * L_s) / ((np.pi / R_ref) * L_c)

        water_mean = water_cut.mean(axis=(0,1))
        sky_mean = sky_cut.mean(axis=(0,1))
        card_mean = card_cut.mean(axis=(0,1))
        print("Calculated mean values per channel")

        water_std = water_cut.std(axis=(0,1))
        sky_std = sky_cut.std(axis=(0,1))
        card_std = card_cut.std(axis=(0,1))

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
        save_to = data_path.parent / (data_path.stem + "_jpeg.csv")
        result.write(save_to, format="ascii.fast_csv")

        print(f"Saved results to {save_to}")
