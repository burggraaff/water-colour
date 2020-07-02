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
from spectacle import io, calibrate, spectral
from spectacle.general import RMS
from astropy import table
from datetime import datetime

# Get the data folder from the command line
path_calibration, path_phone, path_sorad = io.path_from_input(argv)
phone_name = " ".join(path_phone.stem.split("_")[1:-1])

# Find the effective wavelength corresponding to the RGB bands
spectral_response = calibrate.load_spectral_response(path_calibration)
wavelengths_phone = spectral_response[0]
RGB_responses = spectral_response[1:4]
RGB_wavelengths = spectral.effective_wavelengths(wavelengths_phone, RGB_responses)

# SPECTACLE function for effective bandwidth is currently somewhat broken so we
# do it ourselves
RGB_responses_normalised = RGB_responses / RGB_responses.max(axis=1)[:,np.newaxis]
effective_bandwidths = np.trapz(RGB_responses_normalised, x=wavelengths_phone, axis=1)

table_phone = table.Table.read(path_phone)

wavelengths = np.arange(320, 955, 3.3)

def convert_row(row):
    row_split = row.split(";")
    start = row_split[:-1]
    end = np.array(row_split[-1].split(","), dtype=np.float32).tolist()
    row_final = start + end
    return row_final

with open(path_sorad) as file:
    data = file.readlines()
    header = data[0]
    data = data[1:]
    cols = header.split(";")[:-1] + [f"Rrs_{wvl:.1f}" for wvl in wavelengths]

    rows = [convert_row(row) for row in data]
    dtypes = ["S30" for h in header.split(";")[:-3]] + [np.float32, np.float32] + [np.float32 for wvl in wavelengths]

    table_sorad = table.Table(rows=rows, names=cols, dtype=dtypes)

    # Subtract offset
    for wvl in wavelengths:
        table_sorad[f"Rrs_{wvl:.1f}"] = table_sorad[f"Rrs_{wvl:.1f}"] - table_sorad["offset"]

sorad_datetime = [datetime.fromisoformat(DT) for DT in table_sorad["trigger_id"]]
sorad_timestamps = [dt.timestamp() for dt in sorad_datetime]
table_sorad.add_column(table.Column(data=sorad_timestamps, name="UTC"))

def average_row(row):
    Rrs = np.array([row[f"Rrs_{wvl:.1f}"] for wvl in wavelengths])
    RGB_averaged = []

    for response, c in zip(RGB_responses, "RGB"):
        response_interpolated = np.interp(wavelengths, wavelengths_phone, response, left=0, right=0)
        not_nan = np.where(~np.isnan(Rrs))
        Rrs_avg = np.average(Rrs[not_nan], weights=response_interpolated[not_nan])

        RGB_averaged.append(Rrs_avg)

    return RGB_averaged

table_sorad = table_sorad[26000:27500]

RGB_averaged_all = np.array([average_row(row) for row in table_sorad])
RGB_averaged_table = table.Table(data=RGB_averaged_all, names=["Rrs_avg (R)", "Rrs_avg (G)", "Rrs_avg (B)"])
table_sorad = table.hstack([table_sorad, RGB_averaged_table])

data_phone = []
data_sorad = []

for row in table_phone:
    time_differences = np.abs(table_sorad["UTC"] - row["UTC"])
    close_enough = np.where((time_differences <= 60) & (table_sorad["valid"] == "1"))[0]
    if not len(close_enough):
        continue

    print("----")
    Rrs_all = np.array([table_sorad[f"Rrs_{wvl:.1f}"][close_enough] for wvl in wavelengths]).T

    Rrs_mean = Rrs_all.mean(axis=0)
    Rrs_std = Rrs_all.std(axis=0)
    print(np.nanmean(Rrs_std))
    Rrs_err = Rrs_std# / np.sqrt(Rrs_all.shape[0])

    Rrs_avg_all = np.stack([table_sorad[close_enough][f"Rrs_avg ({c})"] for c in "RGB"]).T
    Rrs_avg_mean = Rrs_avg_all.mean(axis=0)
    Rrs_avg_std = Rrs_avg_all.std(axis=0)
    Rrs_avg_err = Rrs_avg_std# / np.sqrt(Rrs_avg_all.shape[0])

    phone_time = datetime.fromtimestamp(row['UTC']).isoformat()
    sorad_time = np.array([datetime.fromtimestamp(time).isoformat() for time in table_sorad[close_enough]["UTC"]])
    print(f"Phone time: {phone_time} ; SoRad time: {sorad_time[0]} -- {sorad_time[-1]}")

    plt.figure(figsize=(3.3,3.3), tight_layout=True)
    for Rrs in Rrs_all:
        plt.plot(wavelengths, Rrs, c="k", alpha=0.2)
    plt.plot(wavelengths, Rrs_mean, c="k")
    for j, c in enumerate("RGB"):
        plt.errorbar(RGB_wavelengths[j], row[f"R_rs ({c})"], xerr=effective_bandwidths[j]/2, yerr=row[f"R_rs_err ({c})"], fmt="o", c=c)
        #plt.errorbar(RGB_wavelengths[j], Rrs_avg_mean[j], xerr=effective_bandwidths[j]/2, yerr=Rrs_avg_err[j], fmt="^", c=c)
    plt.grid(True, ls="--")
    plt.xlim(200, 900)
    plt.xlabel("Wavelength [nm]")
    plt.ylim(0, 0.055)
    plt.ylabel("$R_{rs}$ [sr$^{-1}$]")
    plt.title(f"{phone_name}\n{phone_time}")
    phone_time = phone_time.replace(":", "_")
    plt.savefig(f"SoRad_comparison/{phone_name}_{phone_time}.pdf")
    #plt.show()
    plt.close()

#    data_phone.append(row)
#    data_sorad.append(table_sorad[closest])
#raise Exception
data_phone = table.vstack(data_phone)
data_sorad = table.vstack(data_sorad)

sorad_wavelengths_RGB = [wavelengths[np.abs(wavelengths-wvl).argmin()] for wvl in RGB_wavelengths]

Rrs_phone = np.hstack([data_phone["R_rs (R)"].data, data_phone["R_rs (G)"].data, data_phone["R_rs (B)"].data])
Rrs_sorad_averaged = np.hstack([data_sorad["Rrs_avg (R)"].data, data_sorad["Rrs_avg (G)"].data, data_sorad["Rrs_avg (B)"].data])
rms = RMS(Rrs_phone - Rrs_sorad_averaged)
r = np.corrcoef(Rrs_phone, Rrs_sorad_averaged)[0, 1]

max_val = 0
plt.figure(figsize=(5,5), tight_layout=True)
for j,c in enumerate("RGB"):
    plt.errorbar(data_sorad[f"Rrs_avg ({c})"], data_phone[f"R_rs ({c})"], xerr=0, yerr=data_phone[f"R_rs_err ({c})"], color=c, fmt="o")
    max_val = max(max_val, data_phone[f"R_rs ({c})"].max(), data_sorad[f"Rrs_avg ({c})"].max())
plt.plot([-1, 1], [-1, 1], c='k', ls="--")
plt.xlim(0, 1.05*max_val)
plt.ylim(0, 1.05*max_val)
plt.grid(True, ls="--")
plt.xlabel("SoRad $R_{rs}$ [sr$^{-1}$]")
plt.ylabel(phone_name + " $R_{rs}$ [sr$^{-1}$]")
plt.title(f"$r$ = {r:.2f}     RMS = {rms:.2f} sr$" + "^{-1}$")
plt.savefig(f"comparison_SoRad_X_{phone_name}.pdf")
#plt.show()
