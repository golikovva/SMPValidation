import warnings
from datetime import datetime

import numpy as np
import xarray as xr

from libs.validation import Grid
from libs.validation.datasets.base import (
    Dataset,
    ModelCurrentVelocityDataset,
    ModelDriftDataset,
    ModelEastCurrentDataset,
    ModelNorthCurrentDataset,
    ModelSalinityDataset,
    ModelSicDataset,
    ModelTemperatureDataset,
    ModelThickDataset,
)


class GlorysDataset(Dataset):
    """
    Base class for GLORYS datasets. Handles grid creation and basic dataset functionality.
    """

    def _create_grid(self):
        """
        Creates the grid by extracting latitude and longitude from the dataset files.

        Returns:
            Grid: A Grid object containing latitude and longitude arrays.
        """
        grid_path = sorted(self.path.glob(self._files_template))[0]  # Get the first matching file
        print(f'loading grid from {grid_path}')
        ds = xr.open_dataset(grid_path)  # Load dataset

        # Extract latitude and longitude vectors from the dataset
        lat_vec = ds.coords['latitude'].values
        lon_vec = ds.coords['longitude'].values

        # Create 2D arrays of lat/lon using meshgrid
        lat = np.tile(lat_vec, (lon_vec.size, 1)).T
        lon = np.tile(lon_vec, (lat_vec.size, 1))

        # Create and return a Grid object
        grid = Grid(lat, lon)
        return grid


class GlorysOperativeDataset(GlorysDataset):
    """
    Class for handling GLORYS operative datasets, extending the base GlorysDataset class.
    """

    @property
    def _files_template(self):
        """
        Returns the file template for operative datasets.
        """
        return 'glorys_*/GLORYS_*_00_cmems_mod_glo_phy_anfc_0.083deg_P1D-m.nc'

    @staticmethod
    def _parse_date(file):
        """
        Parses the date from the filename of the dataset.

        Args:
            file (Path): The file object containing the filename.

        Returns:
            datetime.date: The parsed date.
        """
        date_part = file.name.split('_')[1]
        date = datetime.strptime(date_part, '%Y-%m-%d').date()
        return date


class GlorysReanalysisDataset(GlorysDataset):
    """
    Class for handling GLORYS reanalysis datasets.
    """

    def _create_dates_dict(self):
        """
        Creates a dictionary that maps dates to dataset selections.

        Returns:
            dict: Dictionary where keys are dates and values are xarray.Dataset selections.
        """
        result = {}
        for file in self.path.glob(self._files_template):
            ds = xr.open_dataset(file)
            ds_dates_dict = {
                dt.astype('M8[D]').astype('O'): ds.sel(time=[dt])  # Map dates to corresponding dataset slices
                for dt in ds.coords['time'].values
            }
            assert not (result.keys() & ds_dates_dict.keys()), 'files have overlapping dates'
            result.update(ds_dates_dict)
        print(f'parsed {len(result)} dates')
        return result

    def __getitem__(self, date):
        """
        Retrieves data for a specific date, applying averaging and interpolation if necessary.

        Args:
            date (datetime.date): The date for which to retrieve data.

        Returns:
            np.array: The processed data for the specified date.
        """
        if date not in self.dates_dict:
            return None
        ds = self.dates_dict[date]
        item = self._extract_data(ds, load_fn=lambda x: x)  # Extract data from dataset slice
        assert len(item.shape) == 4, f'expected 4D data, got {item.shape}'  # Ensure data is 4D
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            item = np.nanmean(item[self.average_times], axis=0)  # Average over time if specified
        if self.interpolator is not None:
            item = np.stack([self.interpolator(field) for field in item])  # Apply interpolation
        return item

    @property
    def _files_template(self):
        """
        Returns the file template for reanalysis datasets.
        """
        return 'cmems_mod_glo_phy_my*_0.083deg_P1D-m_*.nc'

    @staticmethod
    def _parse_date(file):
        """
        Parses the date from the filename of the reanalysis dataset.

        Args:
            file (Path): The file object containing the filename.

        Raises:
            NotImplementedError: This method should be implemented for reanalysis datasets.
        """
        raise NotImplementedError


class GlorysSicDataset(ModelSicDataset):
    """
    Dataset class for GLORYS sea ice concentration (SIC) data.
    """

    @property
    def _sic_variable(self):
        """
        Specifies the sea ice concentration variable.
        """
        return 'siconc'


class GlorysDriftDataset(ModelDriftDataset):
    """
    Dataset class for GLORYS sea ice drift data.
    """

    def _extract_data(self, file, load_fn=xr.open_dataset):
        """
        Extracts drift data, masking zero values.

        Args:
            file (str): Path to the data file.
            load_fn (callable): Function to load the dataset (default is xarray's open_dataset).

        Returns:
            np.array: The processed drift data.
        """
        data = super(GlorysDriftDataset, self)._extract_data(file, load_fn)
        nan_mask = np.all(data == 0.0, axis=0)  # Mask regions with all zero values
        data[:, nan_mask] = np.nan
        return data

    @property
    def _udrift_variable(self):
        """
        Specifies the u-component drift variable.
        """
        return 'usi'

    @property
    def _vdrift_variable(self):
        """
        Specifies the v-component drift variable.
        """
        return 'vsi'


class GlorysThickDataset(ModelThickDataset):
    """
    Dataset class for GLORYS sea ice thickness data.
    """

    @property
    def _thick_variable(self):
        """
        Specifies the sea ice thickness variable.
        """
        return 'sithick'


class GlorysSalinityDataset(ModelSalinityDataset):
    """
    Dataset class for GLORYS salinity data.
    """

    @property
    def _salinity_variable(self):
        """
        Specifies the salinity variable.
        """
        return 'so'


class GlorysTemperatureDataset(ModelTemperatureDataset):
    """
    Dataset class for GLORYS temperature data.
    """

    @property
    def _temp_variable(self):
        """
        Specifies the temperature variable.
        """
        return 'thetao'


class GlorysEastCurrentDataset(ModelEastCurrentDataset):
    """
    Dataset class for GLORYS eastward current data.
    """

    @property
    def _east_cur_variable(self):
        """
        Specifies the eastward current variable.
        """
        return 'uo'


class GlorysNorthCurrentDataset(ModelNorthCurrentDataset):
    """
    Dataset class for GLORYS northward current data.
    """

    @property
    def _north_cur_variable(self):
        """
        Specifies the northward current variable.
        """
        return 'vo'


class GlorysCurrentVelocityDataset(ModelCurrentVelocityDataset):
    """
    Dataset class for GLORYS current velocity data (combining eastward and northward currents).
    """

    @property
    def _east_cur_variable(self):
        """
        Specifies the eastward current variable.
        """
        return 'uo'

    @property
    def _north_cur_variable(self):
        """
        Specifies the northward current variable.
        """
        return 'vo'


class GlorysOperativeSicDataset(GlorysOperativeDataset, GlorysSicDataset):
    """
    Operative GLORYS sea ice concentration dataset.
    """
    pass


class GlorysReanalysisSicDataset(GlorysReanalysisDataset, GlorysSicDataset):
    """
    Reanalysis GLORYS sea ice concentration dataset.
    """
    pass


class GlorysOperativeDriftDataset(GlorysOperativeDataset, GlorysDriftDataset):
    """
    Operative GLORYS sea ice drift dataset.
    """
    pass


class GlorysReanalysisDriftDataset(GlorysReanalysisDataset, GlorysDriftDataset):
    """
    Reanalysis GLORYS sea ice drift dataset.
    """
    pass


class GlorysOperativeThickDataset(GlorysOperativeDataset, GlorysThickDataset):
    """
    Operative GLORYS sea ice thickness dataset.
    """
    pass


class GlorysReanalysisThickDataset(GlorysReanalysisDataset, GlorysThickDataset):
    """
    Reanalysis GLORYS sea ice thickness dataset.
    """
    pass


class GlorysOperativeSalinityDataset(GlorysOperativeDataset, GlorysSalinityDataset):
    """
    Operative GLORYS salinity dataset.
    """

    @property
    def _files_template(self):
        """
        Returns the file template for operative salinity datasets.
        """
        return 'glorys_*/GLORYS_*_00_cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m.nc'


class GlorysReanalysisSalinityDataset(GlorysReanalysisDataset, GlorysSalinityDataset):
    """
    Reanalysis GLORYS salinity dataset.
    """
    pass


class GlorysOperativeTemperatureDataset(GlorysOperativeDataset, GlorysTemperatureDataset):
    """
    Operative GLORYS temperature dataset.
    """

    @property
    def _files_template(self):
        """
        Returns the file template for operative temperature datasets.
        """
        return 'glorys_*/GLORYS_*_00_cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m.nc'


class GlorysReanalysisTemperatureDataset(GlorysReanalysisDataset, GlorysTemperatureDataset):
    """
    Reanalysis GLORYS temperature dataset.
    """
    pass


class GlorysOperativeEastCurrentDataset(GlorysOperativeDataset, GlorysEastCurrentDataset):
    """
    Operative GLORYS eastward current dataset.
    """

    @property
    def _files_template(self):
        """
        Returns the file template for operative eastward current datasets.
        """
        return 'glorys_*/GLORYS_*_00_cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m.nc'


class GlorysReanalysisEastCurrentDataset(GlorysReanalysisDataset, GlorysEastCurrentDataset):
    """
    Reanalysis GLORYS eastward current dataset.
    """
    pass


class GlorysOperativeNorthCurrentDataset(GlorysOperativeDataset, GlorysNorthCurrentDataset):
    """
    Operative GLORYS northward current dataset.
    """

    @property
    def _files_template(self):
        """
        Returns the file template for operative northward current datasets.
        """
        return 'glorys_*/GLORYS_*_00_cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m.nc'


class GlorysReanalysisNorthCurrentDataset(GlorysReanalysisDataset, GlorysNorthCurrentDataset):
    """
    Reanalysis GLORYS northward current dataset.
    """
    pass


class GlorysOperativeCurrentVelocityDataset(GlorysOperativeDataset, GlorysCurrentVelocityDataset):
    """
    Operative GLORYS current velocity dataset (combining eastward and northward currents).
    """

    @property
    def _files_template(self):
        """
        Returns the file template for operative current velocity datasets.
        """
        return 'glorys_*/GLORYS_*_00_cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m.nc'


class GlorysReanalysisCurrentVelocityDataset(GlorysReanalysisDataset, GlorysCurrentVelocityDataset):
    """
    Reanalysis GLORYS current velocity dataset (combining eastward and northward currents).
    """
    pass


class GlorysOperativeCorrectedSalinityDataset(GlorysOperativeSalinityDataset):
    """
    Operative GLORYS salinity dataset with a correction model applied.
    """

    def __init__(self, path, dst_grid, average_times, name, correction_model):
        """
        Initializes the dataset with an additional correction model.

        Args:
            path (str): Path to the dataset files.
            dst_grid (Grid): Destination grid for interpolation.
            average_times (list): Times to average over.
            name (str): Name of the dataset.
            correction_model (callable): The correction model to apply to the data.
        """
        super().__init__(path, dst_grid, average_times, name)
        self.correction_model = correction_model  # Store the correction model

    def _extract_data(self, file, load_fn=xr.open_dataset):
        """
        Extracts and applies the correction model to the salinity data.

        Args:
            file (str): Path to the data file.
            load_fn (callable): Function to load the dataset.

        Returns:
            np.array: Corrected salinity data.
        """
        ds = load_fn(file)
        field = ds.variables[self._salinity_variable].values  # Get salinity data
        data = self._process_field(field)

        # Calculate day of the year for each time step in the dataset
        days_of_year = [
            (time.astype('datetime64[D]') - time.astype('datetime64[Y]')).astype(int)
            for time in ds.coords['time'].values
        ]

        # Apply the correction model to each time step
        corrected_data = np.stack([
            self.correction_model(data, day_of_year)
            for data, day_of_year in zip(data, days_of_year)
        ])
        return corrected_data  # Return the corrected salinity data
