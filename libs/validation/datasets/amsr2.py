from abc import abstractmethod  # For defining abstract base classes and methods
from datetime import datetime  # For handling date and time operations

import numpy as np  # For numerical operations
import pyproj  # For cartographic projections and transformations
import xarray as xr  # For handling multi-dimensional datasets

from libs.validation import Grid  # Custom Grid class from boreylib library
from libs.validation.datasets.base import Dataset  # Base Dataset class from boreylib

class Amsr2Dataset(Dataset):
    """
    Base class for AMSR JAXA HSI datasets, which defines common grid creation,
    data processing, and extraction methods for handling HSI data files.
    """
    def __init__(self, path, dst_grid=None, average_times=None, name=None, grid_file=None):
        assert grid_file is not None, 'Please specify grid_file argument if using this subclass'
        self.grid_file = grid_file
        super().__init__(path, dst_grid, average_times, name)

    def _create_grid(self, load_fn=xr.open_dataset):
        """
        Creates a stereographic grid with specified latitude and longitude dimensions.

        Returns:
            Grid: A Grid object containing lat/lon data.
        """
        grid_ds = load_fn(self.grid_file)  # Load the grid dataset
        lat, lon = grid_ds.variables['latitude'].values, grid_ds.variables['longitude'].values  # Extract latlon data

        # Create and return a Grid object with the calculated lat/lon values
        grid = Grid(lat, lon)
        return grid

    def _process_field(self, field):
        """
        Processes the raw field data by applying transformations and masking invalid values.

        Args:
            field (np.array): Raw data field to process.

        Returns:
            np.array: Processed data field with time, latitude, and longitude dimensions.
        """
        field = field.astype(np.float32)  # Convert to float32 for consistency
        return field

    def _extract_data(self, file, load_fn=xr.open_dataset):
        """
        Extracts geophysical data from an HDF5 file and applies scaling.

        Args:
            file (str): Path to the data file.
            load_fn (callable): Function to load the dataset (default is xarray's open_dataset).

        Returns:
            np.array: Extracted and processed data array.
        """
        ds = load_fn(file)  # Load the dataset
        entry = ds.variables[self._amsr_variable]  # Extract geophysical data
        data = self._process_field(entry.values)  # Process the raw data
        data = data[None, None, :, :]  # (time, channel, lat, lon)
        return data

    @staticmethod
    def _parse_date(file):
        """
        Parses the date from the filename of the dataset.

        Args:
            file (str): Path to the data file.

        Returns:
            datetime.date: Extracted date from the filename.
        """
        date_part = file.stem.split('_')[1]  # Extract date part from filename
        date = datetime.strptime(date_part, '%Y-%m-%d').date()  # Parse the date
        return date
    
    @property
    @abstractmethod
    def _amsr_variable(self):
        """
        Abstract property to specify the AMSR2 variable name.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

class Amsr2HSIDataset(Amsr2Dataset):
    """
    Dataset class for AMSR JAXA HSI 10km resolution data.
    """

    @property
    def _files_template(self):
        """
        Defines the file naming template for 10km HSI data.

        Returns:
            str: File path template for 10km data.
        """
        return 'amsr2_*/amsr2_*.nc'

    @property
    def _amsr_variable(self):
        """
        Specifies the sea ice concentration variable.
        """
        return 'ASI_Ice_Concentration'