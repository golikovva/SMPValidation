import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from libs.validation import Grid, BarentsKaraGrid
from libs.validation.datasets.base import Dataset


class ShapefileSicDataset(Dataset):
    """
    Class for handling sea ice concentration (SIC) data extracted from shapefiles.
    This dataset is grid-based and converts sea ice grades into percentages.
    """

    def __init__(self, path, resolution=5.0, dst_grid=None, average_times=None, name=None):
        """
        Initializes the ShapefileSicDataset with the specified resolution and parameters.

        Args:
            path (str): Path to the dataset files.
            resolution (float): The grid resolution in kilometers.
            dst_grid (Grid, optional): Destination grid for interpolation.
            average_times (list, optional): Time indices to average over.
            name (str, optional): Name of the dataset.
        """
        self.resolution = resolution  # Set the resolution
        super().__init__(path, dst_grid, average_times, name)  # Initialize the base Dataset class

    def _create_grid(self):
        """
        Creates the grid based on the BarentsKaraGrid class with the specified resolution.

        Returns:
            Grid: A grid object for the Barents and Kara Seas region.
        """
        return BarentsKaraGrid(self.resolution)

    def _process_field(self, field):
        """
        Processes the field by converting sea ice grades to percentages and adding a time dimension.

        Args:
            field (np.array): The raw sea ice concentration data.

        Returns:
            np.array: Processed data with added time dimension.
        """
        field *= 10  # Convert sea ice grades to percentages
        field = field[None, :, :]  # Add time dimension (time, lat, lon)
        return field

    def _extract_data(self, file, load_fn=np.load):
        """
        Extracts data from the .npy file and processes it.

        Args:
            file (str): Path to the .npy file.
            load_fn (callable, optional): Function to load the .npy file. Default is np.load.

        Returns:
            np.array: Extracted and processed data (time, channel, lat, lon).
        """
        field = load_fn(file)  # Load the data
        field = self._process_field(field)  # Process the data
        field = field[:, None, :, :]  # Add channel dimension (time, channel, lat, lon)
        return field

    @staticmethod
    def _parse_date(file):
        """
        Parses the date from the parent folder name of the file.

        Args:
            file (Path): The file object to parse the date from.

        Returns:
            datetime.date: Parsed date.
        """
        date_part = file.parent.name  # Extract the date from the parent folder
        date = datetime.strptime(date_part, '%Y%m%d').date()  # Convert to date object
        return date

    @property
    def _files_template(self):
        """
        Returns the template for matching SIC data files based on the resolution.

        Returns:
            str: File template for SIC data files.
        """
        return f'*/rasterized_S_{int(self.resolution)}km.npy'


class ShapefileDriftDataset(Dataset):
    """
    Class for handling sea ice drift data extracted from shapefiles.
    This dataset loads ice drift data and maps it to specific regions.
    """

    def __init__(self, path, region, dst_grid=None, average_times=None, name=None):
        """
        Initializes the ShapefileDriftDataset with the specified region and parameters.

        Args:
            path (str): Path to the dataset files.
            region (str): The region for which to load the drift data.
            dst_grid (Grid, optional): Destination grid for interpolation.
            average_times (list, optional): Time indices to average over.
            name (str, optional): Name of the dataset.
        """
        self.shp_df = None  # Initialize the shapefile DataFrame to None
        self.region = region  # Set the region
        super().__init__(path, dst_grid, average_times, name)

        # Load the grids mapping for drift data
        with open(self.path / 'grids_mapping.pkl', 'rb') as f:
            self._grids_mapping = pickle.load(f)

    def _create_dates_dict(self):
        """
        Creates a dictionary that maps dates to corresponding ice drift data.

        Returns:
            dict: A dictionary with dates as keys and ice drift data as values.
        """
        self.csv_df = pd.read_csv(self.path / 'ice_drift.csv').dropna()  # Load and clean the CSV file
        result = {
            datetime.strptime(date_str, '%Y-%m-%d').date():
                [self.csv_df[self.csv_df['date'] == date_str]]  # Map dates to corresponding data
            for date_str in self.csv_df['date'].unique()
        }
        print(f'parsed {len(result)} dates')
        return result

    def _create_grid(self):
        """
        Creates the grid for the specific region by loading pre-saved grid data.

        Returns:
            Grid: A grid object for the specified region.
        """
        with open(self.path / 'grids.pkl', 'rb') as f:
            grids = pickle.load(f)
        lat, lon = grids[self.region]  # Get latitude and longitude for the region
        grid = Grid(lat, lon)  # Create a Grid object
        return grid

    def _extract_data(self, file, load_fn=lambda x: x):
        """
        Extracts drift data for the specified region from the dataset.

        Args:
            file (str): Path to the dataset file.
            load_fn (callable): Function to load the dataset.

        Returns:
            np.array: Extracted drift data (time, channels, lat, lon).
        """
        df_date = load_fn(file)  # Load the drift data for the date
        data = np.full((2, *self.src_grid.shape), np.nan)  # Initialize the data array (channels, lat, lon)
        for _, row in df_date.iterrows():
            x = row['x0']
            y = row['y0']
            region, i, j = self._grids_mapping[(x, y)]  # Map coordinates to region
            if region == self.region:
                drift_norm = 51.444 * row['w_drift']  # Convert nm/hr to cm/s
                drift_angle = np.radians(90 - row['d_drift'])  # Convert direction from north to east and to radians
                drift_direction = np.array([np.cos(drift_angle), np.sin(drift_angle)])  # Calculate direction vector
                data[:, i, j] = drift_norm * drift_direction  # Apply drift magnitude and direction
        data = data[None]  # Add time dimension (time, channels, lat, lon)
        return data

    def _process_field(self, field):
        """
        This method should be implemented for field processing.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError

    @staticmethod
    def _parse_date(file):
        """
        This method should be implemented for date parsing.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError

    @property
    def _files_template(self):
        """
        This method should be implemented for providing the file template.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError


class ShapefileThickDataset(Dataset):
    """
    Class for handling sea ice thickness (SIT) data extracted from shapefiles.
    This dataset is grid-based and loads thickness data with the specified resolution.
    """

    def __init__(self, path, resolution=5.0, dst_grid=None, average_times=None, name=None):
        """
        Initializes the ShapefileThickDataset with the specified resolution and parameters.

        Args:
            path (str): Path to the dataset files.
            resolution (float): The grid resolution in kilometers.
            dst_grid (Grid, optional): Destination grid for interpolation.
            average_times (list, optional): Time indices to average over.
            name (str, optional): Name of the dataset.
        """
        self.resolution = resolution  # Set the resolution
        super().__init__(path, dst_grid, average_times, name)

    def _create_grid(self):
        """
        Creates the grid based on the BarentsKaraGrid class with the specified resolution.

        Returns:
            Grid: A grid object for the Barents and Kara Seas region.
        """
        return BarentsKaraGrid(self.resolution)

    def _process_field(self, field):
        """
        Processes the field by adding a time dimension.

        Args:
            field (np.array): The raw sea ice thickness data.

        Returns:
            np.array: Processed data with added time dimension.
        """
        field = field[None, :, :]  # Add time dimension (time, lat, lon)
        return field

    def _extract_data(self, file, load_fn=np.load):
        """
        Extracts data from the .npy file and processes it.

        Args:
            file (str): Path to the .npy file.
            load_fn (callable, optional): Function to load the .npy file. Default is np.load.

        Returns:
            np.array: Extracted and processed data (time, channel, lat, lon).
        """
        field = load_fn(file)  # Load the data
        field = self._process_field(field)  # Process the data
        field = field[:, None, :, :]  # Add channel dimension (time, channel, lat, lon)
        return field

    @staticmethod
    def _parse_date(file):
        """
        Parses the date from the parent folder name of the file.

        Args:
            file (Path): The file object to parse the date from.

        Returns:
            datetime.date: Parsed date.
        """
        date_part = file.parent.name  # Extract the date from the parent folder
        date = datetime.strptime(date_part, '%Y%m%d').date()  # Convert to date object
        return date

    @property
    def _files_template(self):
        """
        Returns the template for matching thickness data files based on the resolution.

        Returns:
            str: File template for thickness data files.
        """
        return f'*/rasterized_thick_{int(self.resolution)}km.npy'
