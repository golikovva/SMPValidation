import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def latlon_to_unit_sphere(lat_arr, lon_arr):
    """
    Convert latitude/longitude (in degrees) to 3D coordinates on a unit sphere.

    Args:
        lat_arr: numpy array or list of latitudes in degrees (shape (...)).
        lon_arr: numpy array or list of longitudes in degrees (shape (...)).

    Returns:
        numpy array of shape (..., 3), where each entry is (x, y, z) on the unit sphere.
    """
    lat_rad = np.deg2rad(lat_arr)
    lon_rad = np.deg2rad(lon_arr)
    cos_lat = np.cos(lat_rad)
    x = cos_lat * np.cos(lon_rad)
    y = cos_lat * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.stack([x, y, z], axis=-1)


class BuoyDataset:
    """
    Dataset for SIMB buoy point data (e.g., ice thickness, snow depth).

    Upon initialization:
      - Reads all CSV files from the `csv_folder`, parses timestamps, and collects
        records (date, latitude, longitude, value).
      - Builds a KDTree in 3D coordinates on a unit sphere (latitude/longitude → (x, y, z))
        for all nodes of the destination grid `dst_grid`.
      - Groups data by date (dict: date → list of (lat, lon, value)).

    When calling ds[date]:
      - Checks if there are any records for that date. If none, returns an array of NaN.
      - Otherwise, for each record (lat, lon, value), converts (lat, lon) → (x, y, z),
        finds the nearest grid node in the 3D KDTree, and accumulates the value in a dictionary
        keyed by the flattened grid index.
      - Averages all values that fell into the same node and returns a 2D array of shape (ny, nx),
        with NaN where no buoy data is available.
    """

    def __init__(
        self,
        csv_folder,
        dst_grid,
        data_variable,
        time_variable="time_stamp (UTC)",
        date_format="%-m/%-d/%y %H:%M",
        print_init=True,
        name="BuoyDataset",
        pattern="*.csv",
        **csv_params
    ):
        """
        Initialize the BuoyDataset.

        Args:
            csv_folder (str or Path):
                Path to the directory containing buoy CSV files. Each CSV file is expected to have
                at least these columns:
                  - "time_stamp (UTC)" (e.g., "10/7/19 0:24")
                  - "latitude" (float)
                  - "longitude" (float)
                  - The `variable` column (e.g., "Ice thickness", "Snow depth").
            dst_grid (Grid):
                A Grid object defining the target latitude/longitude grid. Must have
                two attributes: `dst_grid.lat` and `dst_grid.lon`, both 2D numpy arrays of size (ny, nx).
            variable (str):
                Name of the column in the CSV files to extract (e.g., "Ice thickness").
            date_format (str, optional):
                strptime format to parse "time_stamp (UTC)". Default: "%-m/%-d/%y %H:%M".
                If your platform does not support "%-m"/"%-d", use "%m/%d/%y %H:%M".
            print_init (bool, optional):
                If True, prints how many CSV files were processed and how many unique dates found.

        Raises:
            ValueError: If `csv_folder` is not a directory.
            FileNotFoundError: If no CSV files are found in `csv_folder`.
            RuntimeError: If no valid buoy records are found for the specified `variable`.
        """
        self.name = name
        self.dst_grid = dst_grid
        self.csv_folder = Path(csv_folder)
        if not self.csv_folder.is_dir():
            raise ValueError(f"csv_folder must be a directory, not: {csv_folder!r}")

        self.dst_grid = dst_grid
        self.variable = data_variable
        self.time_variable = time_variable
        self.date_format = date_format

        # Read all CSV files in the folder
        self._raw_records = []  # list of tuples: (date (datetime.date), lat, lon, value)
        csv_files = sorted(self.csv_folder.glob(pattern))
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found in {csv_folder}")

        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, **csv_params)
            except Exception as e:
                warnings.warn(f"Skipping {csv_path.name}: could not read CSV ({e})")
                continue
            try:
                times = pd.to_datetime(
                    df[self.time_variable],
                    format=self.date_format,
                    utc=True,
                    errors="raise"
                )
            except Exception:
                times = pd.to_datetime(df[time_variable], unit='s')

            mask_valid = ~times.isna()          # Drop rows with unparsable timestamps
            df = df.loc[mask_valid].reset_index(drop=True)
            times = times[mask_valid]

            dates = times.dt.date
            latitudes = df["latitude"].astype(float).values
            longitudes = df["longitude"].astype(float).values
            values = df[self.variable].apply(pd.to_numeric, errors='coerce').values

            for dt, lat, lon, val in zip(dates, latitudes, longitudes, values):
                if np.isnan(val).any():
                    continue
                self._raw_records.append((dt, float(lat), float(lon), val))

        if len(self._raw_records) == 0:
            raise RuntimeError(
                f"No valid buoy records found for variable '{self.variable}' in {csv_folder}"
            )

        # Group records by date
        self._records_by_date = defaultdict(list)
        for dt, lat, lon, val in self._raw_records:
            self._records_by_date[dt].append((lat, lon, val))

        # Build 3D KDTree over dst_grid
        self._grid_lat = np.asarray(self.dst_grid.lat)   # shape (ny, nx)
        self._grid_lon = np.asarray(self.dst_grid.lon)   # shape (ny, nx)
        if self._grid_lat.shape != self._grid_lon.shape:
            raise ValueError("dst_grid.lat and dst_grid.lon must have the same shape")

        ny, nx = self._grid_lat.shape
        self._ny, self._nx = ny, nx

        flat_lats = self._grid_lat.ravel()
        flat_lons = self._grid_lon.ravel()
        grid_xyz = latlon_to_unit_sphere(flat_lats, flat_lons)  # shape (ny*nx, 3)

        self._kdtree = cKDTree(grid_xyz)

        if print_init:
            n_dates = len(self._records_by_date)
            n_files = len(csv_files)
            print(f"Initialized BuoyDataset: {n_files} CSV files, "
                  f"{len(self._raw_records)} total records, across {n_dates} dates.")

    def __len__(self):
        """Return the number of unique dates with buoy data."""
        return len(self._records_by_date)

    def __contains__(self, date):
        """Return True if there is buoy data for the given date."""
        return date in self._records_by_date

    def __getitem__(self, date):
        """
        Return a 2D numpy array (ny, nx) for the given datetime.date `date`.

        Each grid node is assigned the nearest buoy value (averaged if multiple
        touched the same node). Nodes with no buoy data are NaN.

        Args:
            date (datetime.date): The date (UTC) to retrieve data for.

        Returns:
            numpy array of shape (ny, nx), dtype float64.
        """
        if date not in self._records_by_date:
            return None

        recs = self._records_by_date[date]
        accum = defaultdict(list)

        for lat, lon, val in recs:
            buoy_xyz = latlon_to_unit_sphere(lat, lon)  # (3,)
            _, idx_flat = self._kdtree.query(buoy_xyz)
            accum[idx_flat].append(val)

        out = np.full((val.size, self._ny * self._nx,), np.nan, dtype=np.float64)
        for idx_flat, vals in accum.items():
            out[:, idx_flat] = np.nanmean(vals, axis=0)

        return out.reshape(val.size, self._ny, self._nx)

    @property
    def dates(self):
        """Return a sorted list of available dates with buoy data."""
        return sorted(self._records_by_date.keys())

    @property
    def grid(self):
        """
        Returns the destination grid if specified, else the source grid.

        Returns:
            Grid: The grid used for this dataset.
        """
        return self.dst_grid
