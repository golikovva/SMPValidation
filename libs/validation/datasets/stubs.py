import datetime

import numpy as np

from libs.validation.datasets.base import Dataset


class ConstantDataset(Dataset):
    """
    A dataset class that returns constant values over a grid for a specified time range.
    Useful for validation or baseline comparisons where the data is uniform across time and space.

    Attributes:
        src_grid (Grid): The source grid on which the constant values are defined.
        value (np.array): The constant value(s) to be returned.
        start_date (datetime.date): The start date for the dataset.
        end_date (datetime.date): The end date for the dataset.
    """

    def __init__(self, grid, value=0.0, name=None,
                 start_date=datetime.date(2000, 1, 1),
                 end_date=datetime.date(2030, 1, 1)):
        """
        Initializes the ConstantDataset with a grid and a constant value.

        Args:
            grid (Grid): The grid on which the constant values are defined.
            value (float or np.array, optional): The constant value to return. Defaults to 0.0.
            name (str, optional): The name of the dataset. Defaults to None.
            start_date (datetime.date, optional): The start date for the dataset. Defaults to January 1, 2000.
            end_date (datetime.date, optional): The end date for the dataset. Defaults to January 1, 2030.
        """
        self.src_grid = grid  # Set the source grid
        self.start_date = start_date  # Set the start date
        self.end_date = end_date  # Set the end date
        super().__init__(path='', dst_grid=None, average_times=None, name=name)

        # Store the constant value as an array
        if isinstance(value, (int, float)):
            self.value = np.array([value])
        else:
            self.value = np.array(value)

    def __getitem__(self, date):
        """
        Returns the constant value for a given date. If the date is out of range, returns None.

        Args:
            date (datetime.date): The date for which to return the constant value.

        Returns:
            np.array or None: The constant value array if the date is within the time range, otherwise None.
        """
        if not (self.start_date <= date < self.end_date):
            return None
        item = np.tile(self.value[:, None, None], self.src_grid.shape)  # Tile the value across the grid
        return item

    def _create_dates_dict(self):
        """
        Creates a dictionary mapping dates to None (indicating no specific data files).

        Returns:
            dict: A dictionary with all dates from start_date to end_date.
        """
        result = {
            self.start_date + datetime.timedelta(days=days): None
            for days in range((self.end_date - self.start_date).days)  # Create a date range
        }
        return result

    def _create_grid(self):
        """
        Returns the source grid.

        Returns:
            Grid: The grid on which the constant values are defined.
        """
        return self.src_grid

    def _process_field(self, field):
        """
        Not implemented for this dataset, as fields are constant values.

        Raises:
            NotImplementedError: This method is not implemented for constant datasets.
        """
        raise NotImplementedError

    def _extract_data(self, file, load_fn=None):
        """
        Not implemented for this dataset, as there is no external data to extract.

        Raises:
            NotImplementedError: This method is not implemented for constant datasets.
        """
        raise NotImplementedError

    @staticmethod
    def _parse_date(file):
        """
        Not implemented for this dataset, as there are no files to parse dates from.

        Raises:
            NotImplementedError: This method is not implemented for constant datasets.
        """
        raise NotImplementedError

    @property
    def _files_template(self):
        """
        Not implemented for this dataset, as there are no files to match.

        Raises:
            NotImplementedError: This method is not implemented for constant datasets.
        """
        raise NotImplementedError


class FusionDataset(Dataset):
    """
    A dataset class that combines data from multiple datasets. This class handles merging data
    from several sources for validation or comparison.

    Attributes:
        datasets (list): A list of datasets to be fused.
    """

    def __init__(self, datasets, dst_grid=None, name=None):
        """
        Initializes the FusionDataset with a list of datasets to combine.

        Args:
            datasets (list): A list of datasets to be combined.
            dst_grid (Grid, optional): The destination grid for interpolation. Defaults to None.
            name (str, optional): The name of the dataset. Defaults to None.
        """
        self.datasets = datasets  # Store the datasets to be fused
        super().__init__(path='', dst_grid=dst_grid, average_times=slice(None), name=name)

    def _create_dates_dict(self):
        """
        Creates a dictionary that maps dates to lists of dataset and date tuples from the input datasets.

        Returns:
            dict: A dictionary with dates as keys and lists of dataset-date tuples as values.
        """
        result = {
            date: [(dataset, date) for dataset in self.datasets if date in dataset.dates_dict]
            for date in set(date for dataset in self.datasets for date in dataset.dates_dict)
        }
        print(f'parsed {len(result)} dates')
        return result

    def _create_grid(self):
        """
        Returns the grid of the first dataset, assuming all datasets share the same grid.

        Returns:
            Grid: The grid shared by all datasets.
        """
        return self.datasets[0].grid  # Assumes all datasets share the same grid

    def _extract_data(self, file, load_fn=lambda x: x):
        """
        Extracts data for a specific date from a specific dataset.

        Args:
            file (tuple): A tuple containing a dataset and a date.
            load_fn (callable, optional): Function to load the dataset. Defaults to identity function.

        Returns:
            np.array: The extracted data for the specified date (time, channels, lat, lon).
        """
        dataset, date = load_fn(file)  # Load the dataset and date
        data = dataset[date][None]  # Add a time dimension (time, channels, lat, lon)
        return data

    def _process_field(self, field):
        """
        Not implemented for this dataset.

        Raises:
            NotImplementedError: This method is not implemented for fusion datasets.
        """
        raise NotImplementedError

    @staticmethod
    def _parse_date(file):
        """
        Not implemented for this dataset.

        Raises:
            NotImplementedError: This method is not implemented for fusion datasets.
        """
        raise NotImplementedError

    @property
    def _files_template(self):
        """
        Not implemented for this dataset.

        Raises:
            NotImplementedError: This method is not implemented for fusion datasets.
        """
        raise NotImplementedError
