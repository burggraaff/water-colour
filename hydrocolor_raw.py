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

# Histograms of raw data
bins = np.linspace(0, camera.saturation, 500)
fig, axs = plt.subplots(nrows=3, figsize=(7,2), tight_layout=True, gridspec_kw={"hspace": 0, "wspace": 0}, sharex=True, sharey=True, squeeze=True)
for ax, data, label in zip(axs, [water_raw, sky_raw, card_raw], ["Water", "Sky", "Grey card"]):
    ax.hist(data.ravel(), bins=bins, color="k")
    ax.set_title(label)
    ax.grid(True, ls="--", alpha=0.7)
axs[-1].set_xlabel("Raw data (ADU)")
plt.show()

# Correct for bias
water_bias, sky_bias, card_bias = calibrate.correct_bias(root, water_raw, sky_raw, card_raw)

# Histograms of corrected data
bins = np.linspace(0, camera.saturation, 500)
fig, axs = plt.subplots(nrows=3, figsize=(7,2), tight_layout=True, gridspec_kw={"hspace": 0, "wspace": 0}, sharex=True, sharey=True, squeeze=True)
for ax, data, label in zip(axs, [water_bias, sky_bias, card_bias], ["Water", "Sky", "Grey card"]):
    ax.hist(data.ravel(), bins=bins, color="k")
    ax.set_title(label)
    ax.grid(True, ls="--", alpha=0.7)
axs[-1].set_xlabel("Bias-corrected data (ADU)")
plt.show()

# Normalising for ISO speed is not necessary since this is a relative measurement

# Dark current is negligible

# Correct for flat-field
water_flat, sky_flat, card_flat = calibrate.correct_flatfield(root, water_bias, sky_bias, card_bias)

# Histograms of flat-fielded data
bins = np.linspace(0, camera.saturation, 500)
fig, axs = plt.subplots(nrows=3, figsize=(7,2), tight_layout=True, gridspec_kw={"hspace": 0, "wspace": 0}, sharex=True, sharey=True, squeeze=True)
for ax, data, label in zip(axs, [water_flat, sky_flat, card_flat], ["Water", "Sky", "Grey card"]):
    ax.hist(data.ravel(), bins=bins, color="k")
    ax.set_title(label)
    ax.grid(True, ls="--", alpha=0.7)
axs[-1].set_xlabel("Flat field-corrected data (a.u.)")
plt.show()

# Demosaick the data
water_RGBG, sky_RGBG, card_RGBG = camera.demosaick(water_flat, sky_flat, card_flat)

# Select the central 100x100 pixels
central_x, central_y = water_RGBG.shape[1]//2, water_RGBG.shape[2]//2
box_size = 100
half_box = box_size // 2
central_slice = np.s_[:, central_x-half_box:central_x+half_box+1, central_y-half_box:central_y+half_box+1]
water_cut = water_RGBG[central_slice]
sky_cut = sky_RGBG[central_slice]
card_cut = card_RGBG[central_slice]

# Histograms of sliced data
bins = np.linspace(0, camera.saturation, 500)
fig, axs = plt.subplots(nrows=3, figsize=(7,2), tight_layout=True, gridspec_kw={"hspace": 0, "wspace": 0}, sharex=True, sharey=True, squeeze=True)
for ax, data, label in zip(axs, [water_cut, sky_cut, card_cut], ["Water", "Sky", "Grey card"]):
    ax.hist(data.ravel(), bins=bins, color="k")
    ax.set_title(label)
    ax.grid(True, ls="--", alpha=0.7)
axs[-1].set_xlabel("Corrected/Selected data (a.u.)")
plt.show()

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

sky_mean = sky_cut.mean(axis=(1,2))
card_mean = card_cut.mean(axis=(1,2))

water_RRS = R_RS(water_cut, sky_mean[:,np.newaxis,np.newaxis], card_mean[:,np.newaxis,np.newaxis])

# Histogram of remote sensing reflectances per channel
xmin, xmax = analyse.symmetric_percentiles(water_RRS)
bins = np.linspace(xmin, xmax, 50)
data_channels = [water_RRS.ravel(), water_RRS[0].ravel(), water_RRS[1::2].ravel(), water_RRS[2].ravel()]
fig, axs = plt.subplots(nrows=4, figsize=(7,4), tight_layout=True, gridspec_kw={"hspace": 0, "wspace": 0}, sharex=True, sharey=True, squeeze=True)
for ax, data, c in zip(axs, data_channels, "krgb"):
    ax.hist(data, bins=bins, color=c)
    ax.grid(True, ls="--", alpha=0.7)
axs[-1].set_xlabel("Remote sensing reflectance [sr$^{-1}$]")
plt.show()

# Calculate the mean R_Rs in the RGB bands
RRS_RGB = [water_RRS[0].ravel(), water_RRS[1::2].ravel(), water_RRS[2].ravel()]
RRS_mean = np.array([R_rs.mean() for R_rs in RRS_RGB])
RRS_std = np.array([R_rs.std() for R_rs in RRS_RGB])

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
    plt.errorbar(RGB_wavelengths[j], RRS_mean[j], xerr=effective_bandwidths[j]/2, yerr=RRS_std[j], c=c, fmt="o")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Remote sensing reflectance [sr$^{-1}$]")
plt.xlim(390, 700)
plt.ylim(ymin=0)
plt.show()
