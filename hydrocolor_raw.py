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

# Get the data folder from the command line
path_water, path_sky, path_card, path_calibration = io.path_from_input(argv)
root = io.find_root_folder(path_calibration)

# Get metadata
camera = io.load_metadata(root)
print("Loaded metadata")

# ISO speed and exposure time are assumed equal between all three images and
# thus can be ignored

# Load data
water_raw = io.load_raw_image(path_water)
sky_raw = io.load_raw_image(path_sky)
card_raw = io.load_raw_image(path_card)
print("Loaded RAW data")

# Correct for bias
water_bias, sky_bias, card_bias = calibrate.correct_bias(root, water_raw, sky_raw, card_raw)
print("Corrected bias")

# Normalising for ISO speed is not necessary since this is a relative measurement

# Dark current is negligible

# Correct for flat-field
water_flat, sky_flat, card_flat = calibrate.correct_flatfield(root, water_bias, sky_bias, card_bias)
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

fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(12,4), tight_layout=True, gridspec_kw={"hspace": 0, "wspace": 0}, sharex="col", sharey="col")

for ax_col, water, sky, card in zip(axs.T, water_all, sky_all, card_all):
    data_combined = np.ravel([water, sky, card])
    xmin, xmax = analyse.symmetric_percentiles(data_combined, percent=0.001)
    bins = np.linspace(xmin, xmax, 100)

    for ax, data in zip(ax_col, [water, sky, card]):
        ax.hist(data.ravel(), bins=bins, color="k")
        ax.set_xlim(xmin, xmax)
        ax.grid(True, ls="--", alpha=0.7)

for ax in axs.ravel():
    ax.tick_params(left=False, labelleft=False)
for ax, label in zip(axs[:,0], ["Water", "Sky", "Grey card"]):
    ax.set_ylabel(label)
for ax, title in zip(axs[0], ["Raw", "Bias-corrected", "Flat-fielded", "Central slice"]):
    ax.set_title(title)

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
spectral_response = calibrate.load_spectral_response(root)
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

# TEMPORARY for SoRad plot

wavelengths = np.arange(320, 955, 3.3)

def convert_row(row):
    row_split = row.split(";")
    start = row_split[:-1]
    end = np.array(row_split[-1].split(","), dtype=np.float32).tolist()
    row_final = start + end
    return row_final

with open("/disks/strw1/burggraaff/hydrocolor/So-Rad_Rrs_Balaton2019.csv") as file:
    data = file.readlines()
    header = data[0]
    data = data[1:]
    cols = header.split(";")[:-1] + [f"Rrs_{wvl:.1f}" for wvl in wavelengths]

    rows = [convert_row(row) for row in data]
    dtypes = ["S30" for h in header.split(";")[:-1]] + [np.float32 for wvl in wavelengths]

    data = table.Table(rows=rows, names=cols, dtype=dtypes)

Rrs = np.array([data[f"Rrs_{wvl:.1f}"][26230] for wvl in wavelengths])

plt.figure(figsize=(3,3), tight_layout=True)
for j, c in enumerate("rgb"):
    plt.errorbar(RGB_wavelengths[j], R_rs[j], xerr=effective_bandwidths[j]/2, yerr=R_rs_err[j], c=c, fmt="o")
plt.plot(wavelengths, Rrs, c='k')
plt.xlabel("Wavelength [nm]")
plt.ylabel("Remote sensing reflectance [sr$^{-1}$]")
plt.xlim(390, 700)
plt.ylim(0, 0.07)
plt.yticks(np.arange(0, 0.071, 0.01))
plt.grid(ls="--")
plt.savefig("comparison_raw.pdf")
plt.show()
