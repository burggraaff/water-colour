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

# Get the data folder from the command line
folder, calibration_folder = io.path_from_input(argv)

# Get metadata
camera = io.load_metadata(calibration_folder)
print("Loaded metadata")

# ISO speed and exposure time are assumed equal between all three images and
# thus can be ignored

# Load data
water_path, sky_path, card_path = [folder/(photo + ".JPG") for photo in ("water", "sky", "greycard")]
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

for ax_col, water, sky, card in zip(axs.T, water_all, sky_all, card_all):
    data_combined = np.ravel([water, sky, card])
    bins = np.linspace(0, 255, 100)

    for ax, data in zip(ax_col, [water, sky, card]):
        ax.hist(data.ravel(), bins=bins, color="k")
        ax.set_xlim(0, 255)
        ax.grid(True, ls="--", alpha=0.7)

for ax, image in zip(axs[:,-1], [water_jpeg, sky_jpeg, card_jpeg]):
    ax.imshow(image)
    ax.tick_params(bottom=False, labelbottom=False)

for ax in axs.ravel():
    ax.tick_params(left=False, labelleft=False)
for ax, label in zip(axs[:,0], ["Water", "Sky", "Grey card"]):
    ax.set_ylabel(label)
for ax, title in zip(axs[0], ["JPEG (full)", "Central slice"]):
    ax.set_title(title)

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
plt.savefig("comparison_jpeg.pdf")
plt.show()
