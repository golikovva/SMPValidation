import warnings
from abc import ABC, abstractmethod  # Abstract base class and abstract method functionality
from pathlib import Path  # To handle file paths

import numpy as np  # For numerical operations
import xarray as xr  # For working with multi-dimensional arrays and netCDF files

from libs.validation.interpolator import Interpolator  # Import custom Interpolator class


class Dataset(ABC):
    """
    Abstract base class representing a dataset. Handles data loading, processing, and interpolation.

    Attributes:
        path (Path): Path to the dataset files.
        dst_grid (Grid): Destination grid for interpolation.
        average_times (list): Times to average over during processing.
        name (str): Name of the dataset.
        dates_dict (dict): Dictionary mapping dates to file paths.
        src_grid (Grid): Source grid containing the data.
        interpolator (Interpolator): Interpolator for regridding the data.
    """

    def __init__(self, path, dst_grid=None, average_times=None, name=None):
        """
        Initializes the Dataset class with the path, grid, and other configurations.

        Args:
            path (str): Path to the dataset files.
            dst_grid (Grid, optional): Destination grid for interpolation.
            average_times (list, optional): Time indices to average over.
            name (str, optional): Name of the dataset.

        Raises:
            ValueError: If dst_grid is provided but average_times is not.
        """
        super().__init__()
        self.path = Path(path)  # Convert path to Path object
        self.dst_grid = dst_grid
        self.average_times = average_times
        self.name = name

        # Ensure average_times is specified when dst_grid is provided
        if self.dst_grid is not None and self.average_times is None:
            raise ValueError('average_times must be specified when dst_grid is '
                             '(4d interpolation is not currently supported)')

        # Optional: Print dataset initialization message
        if self.name is not None:
            print(f'initializing {self.name} dataset')

        self.dates_dict = self._create_dates_dict()  # Create a dictionary of available dates
        self.src_grid = self._create_grid()  # Create the source grid
        self.interpolator = self._create_interpolator()  # Create interpolator if required

    def _create_dates_dict(self):
        """
        Creates a dictionary mapping dates to the corresponding data files.

        Returns:
            dict: A dictionary where keys are dates and values are lists of file paths.
        """
        result = {}
        files = sorted(self.path.glob(self._files_template))  # Find all files matching the template
        for file in files:
            try:
                date = self._parse_date(file)  # Extract date from the file
            except ValueError:
                print(f'skipping {file}')  # Skip files with invalid date formats
                continue
            if date in result:
                result[date].append(file)  # Append file to the date entry if already present
            else:
                result[date] = [file]  # Create new entry for the date
        print(f'parsed {len(result)} dates')
        return result

    def _create_interpolator(self):
        """
        Creates an interpolator if a destination grid is specified.

        Returns:
            Interpolator or None: Interpolator object if dst_grid is provided, else None.
        """
        if self.dst_grid is None:
            return None
        interpolator = Interpolator(self.src_grid, self.dst_grid)  # Initialize interpolator
        print('initializing interpolator')
        interpolator.initialize()  # Initialize interpolator settings
        return interpolator

    def __getitem__(self, date):
        """
        Retrieves and processes the data for a specific date.

        Args:
            date (datetime.date): The date for which to retrieve data.

        Returns:
            np.array: The processed and interpolated data for the specified date.
        """
        if date not in self.dates_dict:
            return None

        result = []
        files = self.dates_dict[date]  # Get files for the specified date
        for file in sorted(files):
            data = self._extract_data(file)  # Extract data from file
            assert len(data.shape) == 4, f'expected 4D data, got {data.shape}'  # Ensure 4D data
            result.append(data)

        result = np.concatenate(result, axis=0)  # Concatenate along the time dimension

        # Average the data over the specified times if provided
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            result = np.nanmean(result[self.average_times], axis=0)  # Apply averaging

        # Apply interpolation if an interpolator is available
        if self.interpolator is not None:
            result = np.stack([self.interpolator(field) for field in result])

        return result

    def __len__(self):
        """
        Returns the number of dates available in the dataset.

        Returns:
            int: The number of dates in the dataset.
        """
        return len(self.dates_dict)

    def __lt__(self, other):
        """
        Implements less-than comparison based on dataset names.

        Args:
            other (Dataset): Another dataset to compare.

        Returns:
            bool: True if this dataset's name is lexicographically less than the other.
        """
        if not isinstance(other, Dataset):
            return NotImplemented
        return self.name < other.name

    @property
    def grid(self):
        """
        Returns the destination grid if specified, else the source grid.

        Returns:
            Grid: The grid used for this dataset.
        """
        if self.dst_grid is not None:
            return self.dst_grid
        return self.src_grid

    @abstractmethod
    def _create_grid(self):
        """
        Abstract method to create the grid for the dataset.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def _process_field(self, field):
        """
        Abstract method to process raw data fields.

        Args:
            field (np.array): Raw data field to process.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def _extract_data(self, file, load_fn=None):
        """
        Abstract method to extract data from a file.

        Args:
            file (str): Path to the data file.
            load_fn (callable, optional): Function to load the file. Defaults to xarray's open_dataset.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _parse_date(file):
        """
        Abstract static method to parse the date from a filename.

        Args:
            file (str): File to parse the date from.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _files_template(self):
        """
        Abstract property that defines the file template for locating dataset files.

        Must be implemented by subclasses.
        """
        raise NotImplementedError


class ModelSicDataset(Dataset):
    """
    Dataset class for handling Sea Ice Concentration (SIC) model data.
    """

    def _process_field(self, field):
        """
        Processes the Sea Ice Concentration (SIC) field by applying masking and unit conversion.

        Args:
            field (np.array): Raw SIC data.

        Returns:
            np.array: Processed SIC data.
        """
        field = np.nan_to_num(field, nan=0.0)  # Replace NaNs with 0.0
        field[:, self.src_grid.land_mask()] = np.nan  # Apply land mask
        field = field * 100  # Convert fraction to percentage
        return field

    def _extract_data(self, file, load_fn=xr.open_dataset):
        """
        Extracts SIC data from the given file.

        Args:
            file (str): Path to the data file.
            load_fn (callable, optional): Function to load the file.

        Returns:
            np.array: Extracted SIC data.
        """
        ds = load_fn(file)
        field = ds.variables[self._sic_variable].values  # Get SIC variable
        data = self._process_field(field)  # Process SIC data
        data = data[:, None, :, :]  # Add channel dimension
        return data

    @property
    @abstractmethod
    def _sic_variable(self):
        """
        Abstract property to specify the SIC variable name.

        Must be implemented by subclasses.
        """
        raise NotImplementedError


class ModelDriftDataset(Dataset):
    """
    Dataset class for handling sea ice drift data.
    """

    def _process_field(self, field):
        """
        Processes the drift data by converting units from m/s to cm/s.

        Args:
            field (np.array): Raw drift data.

        Returns:
            np.array: Processed drift data.
        """
        field = field * 100  # Convert m/s to cm/s
        return field

    def _extract_data(self, file, load_fn=xr.open_dataset):
        """
        Extracts drift data (u and v components) from the given file.

        Args:
            file (str): Path to the data file.
            load_fn (callable, optional): Function to load the file.

        Returns:
            np.array: Extracted drift data.
        """
        ds = load_fn(file)
        ufield = ds.variables[self._udrift_variable].values  # Get u-component
        vfield = ds.variables[self._vdrift_variable].values  # Get v-component
        data = np.stack([
            self._process_field(ufield),
            self._process_field(vfield),
        ], axis=1)  # Stack u and v components
        return data

    @property
    @abstractmethod
    def _udrift_variable(self):
        """
        Abstract property to specify the u-component drift variable name.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _vdrift_variable(self):
        """
        Abstract property to specify the v-component drift variable name.

        Must be implemented by subclasses.
        """
        raise NotImplementedError


class ModelThickDataset(Dataset):
    """
    Dataset class for handling sea ice thickness data.
    """

    def _process_field(self, field):
        """
        Processes the thickness data by applying masking.

        Args:
            field (np.array): Raw thickness data.

        Returns:
            np.array: Processed thickness data.
        """
        field = np.nan_to_num(field, nan=0.0)  # Replace NaNs with 0.0
        field[:, self.src_grid.land_mask()] = np.nan  # Apply land mask
        return field

    def _extract_data(self, file, load_fn=xr.open_dataset):
        """
        Extracts thickness data from the given file.

        Args:
            file (str): Path to the data file.
            load_fn (callable, optional): Function to load the file.

        Returns:
            np.array: Extracted thickness data.
        """
        ds = load_fn(file)
        field = ds.variables[self._thick_variable].values  # Get thickness variable
        data = self._process_field(field)  # Process thickness data
        data = data[:, None, :, :]  # Add channel dimension
        return data

    @property
    @abstractmethod
    def _thick_variable(self):
        """
        Abstract property to specify the ice thickness variable name.

        Must be implemented by subclasses.
        """
        raise NotImplementedError


class ModelSalinityDataset(Dataset):
    """
    Dataset class for handling salinity model data.
    """

    def _process_field(self, field):
        """
        Processes the salinity field without additional transformations.

        Args:
            field (np.array): Raw salinity data.

        Returns:
            np.array: Processed salinity data.
        """
        return field  # No processing required for salinity

    def _extract_data(self, file, load_fn=xr.open_dataset):
        """
        Extracts salinity data from the given file.

        Args:
            file (str): Path to the data file.
            load_fn (callable, optional): Function to load the file.

        Returns:
            np.array: Extracted salinity data.
        """
        ds = load_fn(file)
        field = ds.variables[self._salinity_variable].values  # Get salinity variable
        data = self._process_field(field)  # Process salinity data
        return data

    @property
    @abstractmethod
    def _salinity_variable(self):
        """
        Abstract property to specify the salinity variable name.

        Must be implemented by subclasses.
        """
        raise NotImplementedError


class ModelSurfaceSalinityDataset(ModelSalinityDataset):
    def _extract_data(self, file, load_fn=xr.open_dataset):
        data = super()._extract_data(file, load_fn)
        data = data[:, None, :, :]
        return data


class ModelTemperatureDataset(Dataset):
    """
    Dataset class for handling temperature model data.
    """

    def _process_field(self, field):
        """
        Processes the temperature field without additional transformations.

        Args:
            field (np.array): Raw temperature data.

        Returns:
            np.array: Processed temperature data.
        """
        return field  # No processing required for temperature

    def _extract_data(self, file, load_fn=xr.open_dataset):
        """
        Extracts temperature data from the given file.

        Args:
            file (str): Path to the data file.
            load_fn (callable, optional): Function to load the file.

        Returns:
            np.array: Extracted temperature data.
        """
        ds = load_fn(file)
        field = ds.variables[self._temp_variable].values  # Get temperature variable
        data = self._process_field(field)  # Process temperature data
        return data

    @property
    @abstractmethod
    def _temp_variable(self):
        """
        Abstract property to specify the temperature variable name.

        Must be implemented by subclasses.
        """
        raise NotImplementedError


class ModelEastCurrentDataset(Dataset):
    """
    Dataset class for handling eastward current data.
    """

    def _process_field(self, field):
        """
        Processes the eastward current field without additional transformations.

        Args:
            field (np.array): Raw eastward current data.

        Returns:
            np.array: Processed eastward current data.
        """
        return field  # No processing required for eastward current

    def _extract_data(self, file, load_fn=xr.open_dataset):
        """
        Extracts eastward current data from the given file.

        Args:
            file (str): Path to the data file.
            load_fn (callable, optional): Function to load the file.

        Returns:
            np.array: Extracted eastward current data.
        """
        ds = load_fn(file)
        field = ds.variables[self._east_cur_variable].values  # Get eastward current variable
        data = self._process_field(field)  # Process eastward current data
        return data

    @property
    @abstractmethod
    def _east_cur_variable(self):
        """
        Abstract property to specify the eastward current variable name.

        Must be implemented by subclasses.
        """
        raise NotImplementedError


class ModelNorthCurrentDataset(Dataset):
    """
    Dataset class for handling northward current data.
    """

    def _process_field(self, field):
        """
        Processes the northward current field without additional transformations.

        Args:
            field (np.array): Raw northward current data.

        Returns:
            np.array: Processed northward current data.
        """
        return field  # No processing required for northward current

    def _extract_data(self, file, load_fn=xr.open_dataset):
        """
        Extracts northward current data from the given file.

        Args:
            file (str): Path to the data file.
            load_fn (callable, optional): Function to load the file.

        Returns:
            np.array: Extracted northward current data.
        """
        ds = load_fn(file)
        field = ds.variables[self._north_cur_variable].values  # Get northward current variable
        data = self._process_field(field)  # Process northward current data
        return data

    @property
    @abstractmethod
    def _north_cur_variable(self):
        """
        Abstract property to specify the northward current variable name.

        Must be implemented by subclasses.
        """
        raise NotImplementedError


class ModelCurrentVelocityDataset(Dataset):
    """
    Dataset class for handling combined current velocity (eastward and northward components).
    """

    def _process_field(self, field):
        """
        Processes the current velocity field without additional transformations.

        Args:
            field (np.array): Raw current velocity data.

        Returns:
            np.array: Processed current velocity data.
        """
        return field  # No processing required for current velocity

    def _extract_data(self, file, load_fn=xr.open_dataset):
        """
        Extracts current velocity data (eastward and northward components) from the given file.

        Args:
            file (str): Path to the data file.
            load_fn (callable, optional): Function to load the file.

        Returns:
            np.array: Extracted and combined current velocity data.
        """
        ds = load_fn(file)
        ufield = ds.variables[self._east_cur_variable].values  # Get eastward component
        vfield = ds.variables[self._north_cur_variable].values  # Get northward component
        data = np.stack([
            self._process_field(ufield),
            self._process_field(vfield),
        ], axis=0)  # Stack u and v components along a new axis
        data = np.linalg.norm(data, axis=0)  # Compute the velocity magnitude
        return data

    @property
    @abstractmethod
    def _east_cur_variable(self):
        """
        Abstract property to specify the eastward current variable name.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _north_cur_variable(self):
        """
        Abstract property to specify the northward current variable name.

        Must be implemented by subclasses.
        """
        raise NotImplementedError
