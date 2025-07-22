from datetime import datetime

import xarray as xr

from libs.validation import Grid
from libs.validation.datasets.base import (
    Dataset,
    ModelDriftDataset,
    ModelEastCurrentDataset,
    ModelNorthCurrentDataset,
    ModelSalinityDataset,
    ModelSurfaceSalinityDataset,
    ModelSicDataset,
    ModelTemperatureDataset,
    ModelThickDataset,
)


class NemoDataset(Dataset):
    """
    Base class for handling NEMO model datasets, which provides functionality for
    grid creation and date parsing.
    """

    def _create_grid(self):
        """
        Creates the grid by loading latitude and longitude from the dataset files.

        Returns:
            Grid: A Grid object containing latitude and longitude arrays.
        """
        # Find the first matching file to load the grid information
        grid_path = sorted(self.path.glob(self._files_template))[0]
        print(f'loading grid from {grid_path}')

        # Open the dataset to extract the latitude and longitude
        ds = xr.open_dataset(grid_path)
        lat = ds.variables['nav_lat'].values  # Extract latitude values
        lon = ds.variables['nav_lon'].values  # Extract longitude values

        # Create and return a Grid object
        grid = Grid(lat, lon)
        return grid

    @property
    def _files_template(self):
        """
        Returns the file template pattern for locating NEMO model dataset files.

        Returns:
            str: The file path template.
        """
        return 'run_*/NESTP12-VP1_*_forecast.1h_icemod.nc'

    @staticmethod
    def _parse_date(file):
        """
        Parses the date from the filename of the NEMO dataset files.

        Args:
            file (Path): The file object containing the filename.

        Returns:
            datetime.date: The parsed date from the filename.
        """
        # Extract the date part from the filename and parse it
        date_part = file.name.split('_')[1]
        date = datetime.strptime(date_part, 'y%Ym%md%d').date()  # Parse as 'yYYYYmMMdDD'
        return date


class NemoSicDataset(ModelSicDataset, NemoDataset):
    """
    Dataset class for handling NEMO sea ice concentration (SIC) data.
    """

    @property
    def _sic_variable(self):
        """
        Specifies the sea ice concentration variable.

        Returns:
            str: The variable name for sea ice concentration.
        """
        return 'siconc'


class NemoDriftDataset(ModelDriftDataset, NemoDataset):
    """
    Dataset class for handling NEMO sea ice drift data.
    """

    @property
    def _udrift_variable(self):
        """
        Specifies the u-component drift variable.

        Returns:
            str: The variable name for the u-component of sea ice drift.
        """
        return 'sivelu'

    @property
    def _vdrift_variable(self):
        """
        Specifies the v-component drift variable.

        Returns:
            str: The variable name for the v-component of sea ice drift.
        """
        return 'sivelv'


class NemoThickDataset(ModelThickDataset, NemoDataset):
    """
    Dataset class for handling NEMO sea ice thickness data.
    """

    @property
    def _thick_variable(self):
        """
        Specifies the sea ice thickness variable.

        Returns:
            str: The variable name for sea ice thickness.
        """
        return 'sithic'


class NemoSalinityDataset(ModelSalinityDataset, NemoDataset):
    """
    Dataset class for handling NEMO salinity data.
    """

    @property
    def _files_template(self):
        """
        Returns the file template for locating NEMO salinity data files.

        Returns:
            str: The file path template for salinity data.
        """
        return 'run_*/NESTP12-VP1_*_forecast.*_gridT.nc'

    @property
    def _salinity_variable(self):
        """
        Specifies the salinity variable.

        Returns:
            str: The variable name for salinity data.
        """
        return 'vosaline'
    
class NemoSurfaceSalinityDataset(ModelSurfaceSalinityDataset, NemoDataset):
    """
    Dataset class for handling NEMO salinity data.
    """

    @property
    def _files_template(self):
        """
        Returns the file template for locating NEMO salinity data files.

        Returns:
            str: The file path template for salinity data.
        """
        return 'run_*/NESTP12-VP1_*_forecast.*_gridTsurf.nc'

    @property
    def _salinity_variable(self):
        """
        Specifies the salinity variable.

        Returns:
            str: The variable name for salinity data.
        """
        return 'sosaline'


class NemoTemperatureDataset(ModelTemperatureDataset, NemoDataset):
    """
    Dataset class for handling NEMO temperature data.
    """

    @property
    def _files_template(self):
        """
        Returns the file template for locating NEMO temperature data files.

        Returns:
            str: The file path template for temperature data.
        """
        return 'run_*/NESTP12-VP1_*_forecast.*_gridT*.nc'

    @property
    def _temp_variable(self):
        """
        Specifies the temperature variable.

        Returns:
            str: The variable name for temperature data.
        """
        return 'votemper'


class NemoEastCurrentDataset(ModelEastCurrentDataset, NemoDataset):
    """
    Dataset class for handling NEMO eastward current data.
    """

    @property
    def _files_template(self):
        """
        Returns the file template for locating NEMO eastward current data files.

        Returns:
            str: The file path template for eastward current data.
        """
        return 'run_*/NESTP12-VP1_*_forecast.*_gridV*.nc'

    @property
    def _east_cur_variable(self):
        """
        Specifies the eastward current variable.

        Returns:
            str: The variable name for eastward current data.
        """
        return 'vozocrtx'


class NemoNorthCurrentDataset(ModelNorthCurrentDataset, NemoDataset):
    """
    Dataset class for handling NEMO northward current data.
    """

    @property
    def _files_template(self):
        """
        Returns the file template for locating NEMO northward current data files.

        Returns:
            str: The file path template for northward current data.
        """
        return 'run_*/NESTP12-VP1_*_forecast.*_gridV*.nc'

    @property
    def _north_cur_variable(self):
        """
        Specifies the northward current variable.

        Returns:
            str: The variable name for northward current data.
        """
        return 'vomecrty'
