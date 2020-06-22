import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, calibrate, spectral
from spectacle.general import RMS
from astropy import table
from datetime import datetime

path_ed = io.Path("/disks/strw1/burggraaff/hydrocolor/SoRad/So-Rad_Ed_Balaton2019.csv")
path_ls = io.Path("/disks/strw1/burggraaff/hydrocolor/SoRad/So-Rad_Ls_Balaton2019.csv")
path_lt = io.Path("/disks/strw1/burggraaff/hydrocolor/SoRad/So-Rad_Lt_Balaton2019.csv")
path_calibration = io.Path("/disks/strw1/burggraaff/SPECTACLE_data/iPhone_SE/")

wavelengths_sorad = np.arange(320, 955, 3.3)

def convert_row(row):
    row_split = row.split(";")
    start = row_split[:-1]
    end = np.array(row_split[-1].split(","), dtype=np.float32).tolist()
    row_final = start + end
    return row_final

def read_sorad_csv(path_sorad, label, wavelengths=wavelengths_sorad):
    with open(path_sorad) as file:
        data = file.readlines()
        header = data[0]
        data = data[1:]
        cols = header.split(";")[:-1] + [f"{label}_{wvl:.1f}" for wvl in wavelengths]

        rows = [convert_row(row) for row in data]
        dtypes = ["S30" for h in header.split(";")[:-1]] + [np.float32 for wvl in wavelengths]

        table_sorad = table.Table(rows=rows, names=cols, dtype=dtypes)

        sorad_datetime = [datetime.fromisoformat(DT) for DT in table_sorad["trigger_id"]]
        sorad_timestamps = [dt.timestamp() for dt in sorad_datetime]
        table_sorad.add_column(table.Column(data=sorad_timestamps, name="UTC"))

        table_sorad = table_sorad[:]

        return table_sorad

# Read the (ir)radiance data and combine them into one big table
table_ed, table_ls, table_lt = [read_sorad_csv(path, label) for path, label in zip([path_ed, path_ls, path_lt], ["Ed", "Ls", "Lt"])]
print("Loaded Ed, Ls, Lt tables")
table_combined = table.join(table_ed, table_ls, keys=table_ed.keys()[:19]+["UTC"])
table_combined = table.join(table_combined, table_lt, keys=table_ed.keys()[:19]+["UTC"])
print("Combined tables")

# Calculate hyperspectral R_rs
print("Calculating hyperspectral R_rs...")
for wvl in wavelengths_sorad:
    Rrs = (table_combined[f"Lt_{wvl:.1f}"] - 0.028 * table_combined[f"Ls_{wvl:.1f}"]) / table_combined[f"Ed_{wvl:.1f}"]
    Rrs_col = table.Column(data=Rrs, name=f"Rrs_{wvl:.1f}")
    table_combined.add_column(Rrs_col)
    print(f"{wvl:.1f}", end=" ")
print()

# Find the effective wavelength corresponding to the RGB bands
spectral_response = calibrate.load_spectral_response(path_calibration)
wavelengths = spectral_response[0]
RGB_responses = spectral_response[1:4]
RGB_wavelengths = spectral.effective_wavelengths(wavelengths, RGB_responses)

def integrate_spectrum(spectrum, response):
    response_interpolated = np.interp(wavelengths_sorad, wavelengths, response, left=0, right=0)
    spectrum_normalised = spectrum * response_interpolated
    not_nan = np.where(~np.isnan(spectrum_normalised))
    spectrum_integrated = np.trapz(spectrum_normalised[not_nan], x=wavelengths_sorad[not_nan])
    return spectrum_integrated

def integrate_row(row):
    Ed  = np.array([row[f"Ed_{wvl:.1f}"] for wvl in wavelengths_sorad])
    Ls  = np.array([row[f"Ls_{wvl:.1f}"] for wvl in wavelengths_sorad])
    Lt  = np.array([row[f"Lt_{wvl:.1f}"] for wvl in wavelengths_sorad])

    RGB_integrated = []

    for response, c in zip(RGB_responses, "RGB"):
        Ed_int = integrate_spectrum(Ed, response)
        Ls_int = integrate_spectrum(Ls, response)
        Lt_int = integrate_spectrum(Lt, response)
        Rrs_int = (Lt_int - 0.028 * Ls_int) / Ed_int

        RGB_integrated.extend([Ed_int, Ls_int, Lt_int, Rrs_int])

    return RGB_integrated

RGB_integrated_all = np.array([integrate_row(row) for row in table_combined])
RGB_integrated_table = table.Table(data=RGB_integrated_all, names=["Ed (R)", "Ls (R)", "Lt (R)", "Rrs (R)", "Ed (G)", "Ls (G)", "Lt (G)", "Rrs (G)", "Ed (B)", "Ls (B)", "Lt (B)", "Rrs (B)"])

table_combined = table.hstack([table_combined, RGB_integrated_table])

def average_row(row):
    Rrs = np.array([row[f"Rrs_{wvl:.1f}"] for wvl in wavelengths_sorad])
    RGB_averaged = []

    for response, c in zip(RGB_responses, "RGB"):
        response_interpolated = np.interp(wavelengths_sorad, wavelengths, response, left=0, right=0)
        not_nan = np.where(~np.isnan(Rrs))
        Rrs_avg = np.average(Rrs[not_nan], weights=response_interpolated[not_nan])

        RGB_averaged.append(Rrs_avg)

    return RGB_averaged

RGB_averaged_all = np.array([average_row(row) for row in table_combined])
RGB_averaged_table = table.Table(data=RGB_averaged_all, names=["Rrs_avg (R)", "Rrs_avg (G)", "Rrs_avg (B)"])
table_combined = table.hstack([table_combined, RGB_averaged_table])

# Plot an example R_rs spectrum
i = 236
Rrs = [table_combined[i][f"Rrs_{wvl:.1f}"] for wvl in wavelengths_sorad]
plt.figure(figsize=(3,3), tight_layout=True)
plt.plot(wavelengths_sorad, Rrs, c='k')
for wvl, c in zip(RGB_wavelengths, "RGB"):
    plt.scatter(wvl, table_combined[i][f"Rrs ({c})"], c=c)
    plt.scatter(wvl, table_combined[i][f"Rrs_avg ({c})"], c=c, marker="^")
plt.xlabel("Wavelength (nm)")
plt.ylabel("$R_{rs}$ [sr$^{-1}$]")
plt.xlim(390, 700)
plt.ylim(ymin=0)
plt.grid(ls="--")
plt.show()

max_val = 0
plt.figure(figsize=(4,4), tight_layout=True)
for c in "RGB":
    plt.scatter(table_combined[f"Rrs ({c})"], table_combined[f"Rrs_avg ({c})"], c=c)
    max_val = max(max_val, table_combined[f"Rrs ({c})"].max(), table_combined[i][f"Rrs_avg ({c})"].max())
    rms = RMS(table_combined[f"Rrs ({c})"] - table_combined[f"Rrs_avg ({c})"])
    rms_rel = RMS((table_combined[f"Rrs ({c})"] - table_combined[f"Rrs_avg ({c})"])/table_combined[f"Rrs ({c})"])
    print(f"{c} band: RMS = {rms:.4f} ; relative RMS = {100*rms_rel:.1f} %")
plt.plot([-10, 10], [-10, 10], c='k', ls="--")
#plt.xticks(np.arange(0, 0.2, 0.02))
#plt.yticks(np.arange(0, 0.2, 0.02))
plt.xlim(0, 1.05*max_val)
plt.ylim(0, 1.05*max_val)
plt.grid(ls="--")
plt.xlabel("$R_{rs}$ [sr$^{-1}$] (integrated)")
plt.ylabel("$R_{rs}$ [sr$^{-1}$] (averaged)")
plt.show()

# Correlation plot between broadband and the closest wavelength - see if B and G are always below