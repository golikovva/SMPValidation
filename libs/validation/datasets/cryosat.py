from datetime import datetime  # For handling and manipulating date and time objects

import xarray as xr  # For working with labeled multi-dimensional arrays (NetCDF format)

from libs.validation import Grid  # Import custom Grid class from boreylib
from libs.validation.datasets.base import ModelThickDataset  # Import base class for thickness datasets


class CryosatThickDataset(ModelThickDataset):
    """
    Class to handle CryoSat sea ice thickness dataset, inheriting from the ModelThickDataset base class.

    This class manages grid creation, data extraction, and parsing dates from CryoSat files.
    """

    def __init__(self, path, dst_grid=None, average_times=None, name=None):
        """
        Initializes the CryosatThickDataset object.

        Args:
            path (str): The path to the CryoSat dataset.
            dst_grid (Grid, optional): The destination grid for interpolation (optional).
            average_times (list, optional): Time indices for averaging (optional).
            name (str, optional): Name of the dataset (optional).
        """
        super().__init__(path, dst_grid, average_times, name)  # Call the base class constructor

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
        lat = ds.variables['lat'].values  # Extract latitude values
        lon = ds.variables['lon'].values  # Extract longitude values

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
        parts = file.name.split('_')
        start_date_str = parts[8]  # Extract the start date as string
        end_date_str = parts[9]  # Extract the end date as string

        # Convert string dates to datetime objects
        start_date = datetime.strptime(start_date_str, "%Y%m%d")
        end_date = datetime.strptime(end_date_str, "%Y%m%d")

        # Calculate the middle date between start and end
        middle_date = start_date + (end_date - start_date) / 2
        return middle_date.date()  # Return the date part

    @property
    def _files_template(self):
        """
        Provides the file template pattern for locating CryoSat sea ice thickness data files.

        Returns:
            str: The template string for file paths.
        """
        return 'SEAICE_ARC_PHY_L4_NRT_011_014/esa_obs-si_arc_phy-sit_nrt_l4_multi_P1D-m_202207/**/*.nc'

    @property
    def _thick_variable(self):
        """
        Specifies the variable name for sea ice thickness data.

        Returns:
            str: The variable name for sea ice thickness.
        """
        return 'analysis_sea_ice_thickness'  # The variable name in the dataset for sea ice thickness
