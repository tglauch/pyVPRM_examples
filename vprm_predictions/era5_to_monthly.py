"""separate annual ERA5 netCDF files into monthly"""

import socket
from pathlib import Path

import pandas as pd
import xarray as xr

from loguru import logger

ERA5_DATA_DIR = Path("/mnt/amp/CarbonWatchUrban/urbanVPRM/ERA5/")
if socket.gethostname() == "hutl21264.gns.cri.nz":
    # prepend /home/timh to path for laptop
    ERA5_DATA_DIR = Path("/home/timh", *ERA5_DATA_DIR.parts[1:])

if __name__ == "__main__":
    this_year = 2023
    for this_year in range(2020, 2024):
        for this_month in range(1, 13):
            fname = f"ERA5_2mT_msdswrf_ssrd_NZ_{this_year}.nc"
            xds = xr.open_dataset(Path(ERA5_DATA_DIR, fname))
            month = pd.to_datetime(xds["time"]).month

            fname = Path(
                ERA5_DATA_DIR
                / "monthly"
                / f"ERA5_2mT_msdswrf_ssrd_NZ_{this_year}_{this_month:02d}.nc"
            )
            logger.info(f"writing {fname}")
            # xds.sel(time=(month == this_month)).to_netcdf(fname, mode="w")
