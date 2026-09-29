from datetime import datetime  # For handling and manipulating date and time objects

import xarray as xr  # For working with labeled multi-dimensional arrays (NetCDF format)
import numpy as np  # For numerical operations on arrays
from libs.validation import Grid  # Import custom Grid class from boreylib
from libs.validation.datasets.base import ModelThickDataset  # Import base class for thickness datasets
from libs.validation.grid_utils import grid_lat_lon_2d  # Import utility function to extract lat/lon from grid

class CryosatThickDataset(ModelThickDataset):
    """
    Class to handle CryoSat sea ice thickness dataset, inheriting from the ModelThickDataset base class.

    This class manages grid creation, data extraction, and parsing dates from CryoSat files.
    """

    def __init__(
        self, path, dst_grid=None, average_times=None, name=None, lats_slice=None,
        *, files_template=None, interpolation_cache_dir=None,
    ):
        """
        Initializes the CryosatThickDataset object.

        Args:
            path (str): The path to the CryoSat dataset.
            dst_grid (Grid, optional): The destination grid for interpolation (optional).
            average_times (list, optional): Time indices for averaging (optional).
            name (str, optional): Name of the dataset (optional).
            interpolation_cache_dir (str or Path, optional): Directory for reusable
                interpolation weights. None disables the disk cache.
        """
        if lats_slice is None:
            lats_slice = slice(-300, None)
        self.lats_slice = lats_slice

        super().__init__(
            path, dst_grid, average_times, name, files_template=files_template,
            interpolation_cache_dir=interpolation_cache_dir,
        )  # Call the base class constructor

    def _create_grid(self):
        """
        Creates the grid by loading latitude and longitude from the dataset files.

        Returns:
            Grid: A Grid object containing latitude and longitude arrays.
        """
        # Find the first matching file for loading the grid
        grid_path = sorted(self.path.glob(self._files_template))[0]
        print(f'loading grid from {grid_path}')

        # Load the dataset to extract latitude and longitude
        ds = xr.open_dataset(grid_path)
        lat, lon = grid_lat_lon_2d(ds)
        # lat = ds['latitude'].values[self.lats_slice]  # Extract latitude values
        # lon = ds['longitude'].values  # Extract longitude values
        # lat, lon = np.meshgrid(lat, lon, indexing='ij')  # Create 2D meshgrid

        # Create a Grid object with the extracted latitude and longitude
        grid = Grid(lat, lon)
        return grid

    @staticmethod
    def _parse_date(file):
        """
        Parses the date from the filename by calculating the middle date between start and end dates.

        Args:
            file (Path): The file object containing the filename.

        Returns:
            datetime.date: The middle date between start and end dates.
        """
        # Extract date parts from the filename
        dates = file.name.split('_')[-1]
        date = '-'.join(dates.split('-')[:3])

        # Convert string dates to datetime objects
        date = datetime.strptime(date, "%Y-%m-%d").date()
        return date
    
    def _process_field(self, field):
        """
        Processes the thickness data by applying masking.

        Args:
            field (np.array): Raw thickness data.

        Returns:
            np.array: Processed thickness data.
        """
        # field = field[:, self.lats_slice]
        field = np.nan_to_num(field, nan=0.0)  # Replace NaNs with 0.0
        field[:, self.src_grid.land_mask()] = np.nan  # Apply land mask
        return field

    @property
    def _default_files_template(self) -> str:
        """
        Provides the file template pattern for locating CryoSat sea ice thickness data files.

        Returns:
            str: The template string for file paths.
        """
        return 'esa_obs-si_arc_phy-sit_nrt_l4-*.nc'  # Template for CryoSat thickness data files

    @property
    def _thick_variable(self):
        """
        Specifies the variable name for sea ice thickness data.

        Returns:
            str: The variable name for sea ice thickness.
        """
        return 'sea_ice_thickness'  # The variable name in the dataset for sea ice thickness
