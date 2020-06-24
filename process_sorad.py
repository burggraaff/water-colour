import numpy as np
from matplotlib import pyplot as plt
from astropy import table
from sys import argv

folder = argv[1]

wavelengths = np.arange(320, 955, 3.3)

def convert_row(row):
    row_split = row.split(";")
    start = row_split[:-1]
    end = np.array(row_split[-1].split(","), dtype=np.float32).tolist()
    row_final = start + end
    return row_final

with open(f"{folder}/So-Rad_Rrs_Balaton2019.csv") as file:
    data = file.readlines()
    header = data[0]
    data = data[1:]
    cols = header.split(";")[:-1] + [f"Rrs_{wvl:.1f}" for wvl in wavelengths]

    rows = [convert_row(row) for row in data]
    dtypes = ["S30" for h in header.split(";")[:-1]] + [np.float32 for wvl in wavelengths]

    data = table.Table(rows=rows, names=cols, dtype=dtypes)


Rrs = np.array([[row[f"Rrs_{wvl:.1f}"] for wvl in wavelengths] for row in data[::100]]).T

plt.figure(figsize=(6,3), tight_layout=True)
plt.plot(wavelengths, Rrs, c="k", alpha=0.1)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Remote sensing reflectance [sr$^{-1}$]")
plt.xlim(320, 955)
plt.ylim(0, 0.1)
plt.yticks(np.arange(0, 0.12, 0.02))
plt.grid(ls="--")
plt.show()
