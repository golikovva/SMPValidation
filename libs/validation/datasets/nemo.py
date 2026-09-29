from datetime import datetime

import xarray as xr
import numpy as np
from libs.validation import Grid
from libs.validation.datasets.base import (
    Dataset,
    ModelDriftDataset,
    ModelEastCurrentDataset,
    ModelNorthCurrentDataset,
    ModelCurrentDataset,
    ModelSalinityDataset,
    ModelSurfaceSalinityDataset,
    ModelSicDataset,
    ModelTemperatureDataset,
    ModelSurfaceTemperatureDataset,
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
    def _default_files_template(self) -> str:
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
    """NEMO sea-ice thickness over the ice-covered part of each cell.

    ``thickness_source="sithic"`` reads thickness directly (the default).
    ``thickness_source="sivolu"`` derives thickness as ``sivolu / siconc``
    for each time sample, before time averaging or interpolation. In this
    mode, ``sivolu`` must be ice volume per cell area in metres, and
    ``siconc`` must be a fraction in [0, 1], not a percentage.

    Ice-free cells, missing/invalid values and land remain NaN. If siconc
    is available in direct mode, it is also used to mask ice-free cells.
    """

    def __init__(
        self,
        path,
        dst_grid=None,
        average_times=None,
        name=None,
        *,
        files_template=None,
        thickness_source="sithic",
        interpolation_cache_dir=None,
    ):
        if thickness_source not in ("sithic", "sivolu"):
            raise ValueError("thickness_source must be 'sithic' or 'sivolu'")
        self.thickness_source = thickness_source
        super().__init__(
            path, dst_grid, average_times, name, files_template=files_template,
            interpolation_cache_dir=interpolation_cache_dir,
        )

    def _process_field(self, field):
        """Preserve missing values and mask nonpositive thickness and land."""
        field = np.where(np.isfinite(field) & (field > 0), field, np.nan)
        field[:, self.src_grid.land_mask()] = np.nan
        return field

    def _extract_data(self, file, load_fn=xr.open_dataset):
        """Read or derive thickness with shape (time, 1, y, x)."""
        with load_fn(file) as ds:
            if self.thickness_source == "sivolu":
                field = ds["sivolu"]
                concentration = ds["siconc"]
            else:
                field = ds[self._thick_variable]
                concentration = ds.get("siconc")

            if concentration is not None:
                valid_ice = (
                    np.isfinite(concentration)
                    & (concentration > 0)
                    & (concentration <= 1)
                )
                if self.thickness_source == "sivolu":
                    field = field / concentration.where(valid_ice)
                else:
                    field = field.where(valid_ice)

            data = self._process_field(field.values)

        return data[:, None, :, :]

    @property
    def _thick_variable(self):
        """Name of the variable used for directly stored ice thickness."""
        return 'sithic'


class NemoSalinityDataset(ModelSalinityDataset, NemoDataset):
    """
    Dataset class for handling NEMO salinity data.
    """

    @property
    def _default_files_template(self) -> str:
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
    def _default_files_template(self) -> str:
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
    def _default_files_template(self) -> str:
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

class NemoSurfaceTemperatureDataset(ModelSurfaceTemperatureDataset, NemoDataset):
    """
    Dataset class for handling NEMO temperature data.
    """

    @property
    def _default_files_template(self) -> str:
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
        return 'sosstsst'

class NemoEastCurrentDataset(ModelEastCurrentDataset, NemoDataset):
    """
    Dataset class for handling NEMO eastward current data.
    """

    @property
    def _default_files_template(self) -> str:
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
    def _default_files_template(self) -> str:
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
    

class NemoCurrentDataset(ModelCurrentDataset, NemoDataset):
    """
    Dataset class for handling NEMO sea ice current data.
    """

    @property
    def _default_files_template(self) -> str:
        """
        Returns the file template for locating NEMO northward current data files.

        Returns:
            str: The file path template for northward current data.
        """
        return 'run_*/NESTP12-VP1_*_forecast.*_gridV*.nc'
    
    @property
    def _east_cur_variable(self):
        """
        Specifies the u-component current variable.

        Returns:
            str: The variable name for the u-component of sea ice current.
        """
        return 'vozocrtx'

    @property
    def _north_cur_variable(self):
        """
        Specifies the v-component current variable.

        Returns:
            str: The variable name for the v-component of sea ice current.
        """
        return 'vomecrty'
    


class NemoGeneralIceDataset(NemoDataset):
    def __init__(
        self,
        path,
        variables,
        dst_grid=None,
        average_times=None,
        name=None,
        mask_var=None,
        *,
        files_template=None,
        interpolation_cache_dir=None,
    ):
        self.variables=variables
        self.mask_var=mask_var
        super().__init__(
            path, dst_grid, average_times, name, files_template=files_template,
            interpolation_cache_dir=interpolation_cache_dir,
        )

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
        data = []
        for variable in self.variables:
            if self.mask_var is not None:
                values = ds.variables[variable].values * ds.variables[self.mask_var].values
            else:
                values = ds.variables[variable].values 
            data.append(values)
        data = np.stack([self._process_field(field) for field in data], axis=1)
        return data
