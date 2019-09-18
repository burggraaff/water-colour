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
from spectacle import io
from astropy import table
from datetime import datetime
from os import walk

# Get the data folder from the command line
pattern, *folders = io.path_from_input(argv)

all_data = []
for folder in folders:
    for tup in walk(folder):
        folder_here = io.Path(tup[0])
        file_path = folder_here / pattern

        if file_path.exists():
            print(file_path)
            data_here = table.Table.read(file_path)
            all_data.append(data_here)

        else:
            continue

data_combined = table.vstack(all_data)
data_combined.sort("UTC")

print(data_combined)
