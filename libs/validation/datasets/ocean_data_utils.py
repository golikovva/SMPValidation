from datetime import datetime

import xarray as xr
import numpy as np
from libs.validation import Grid
from libs.validation.datasets.data_utils import (
    NCs2sDataset
)


class TopazOpDataset(NCs2sDataset):
    @staticmethod
    def _parse_date(file):
        date_part = file.stem.split('_')[-1]
        date = np.datetime64(date_part)
        return date

    @property
    def _files_template(self):
        return '*cmems_mod_arc_phy_anfc*'

    def _create_grid(self):
        grid_path = sorted(self.path.glob(self._files_template))[0]
        print(f'loading grid from {grid_path}')
        with xr.open_dataset(grid_path, cache=False) as ds:
            lon = ds.coords['longitude'].values
            lat = ds.coords['latitude'].values
            lat, lon = np.meshgrid(lat, lon)
            grid = {'longitude': lon.T, 'latitude': lat.T}
        return grid

    @property
    def _file_len(self):
        return 'D'


class NeXtSIMDataset(NCs2sDataset):
    @staticmethod
    def _parse_date(file):
        date_part = file.stem.split('_')[-1]
        date = np.datetime64(date_part)
        return date

    @property
    def _files_template(self):
        return '*cmems_mod_arc_phy_anfc_nextsim*'

    def _create_grid(self):
        grid_path = sorted(self.path.glob(self._files_template))[0]
        print(f'loading grid from {grid_path}')
        with xr.open_dataset(grid_path, cache=False) as ds:
            lon = ds.coords['longitude'].values
            lat = ds.coords['latitude'].values
            lat, lon = np.meshgrid(lat, lon)
            grid = {'longitude': lon.T, 'latitude': lat.T}
        return grid

    @property
    def _file_len(self):
        return 'D'

class OrasDataset(NCs2sDataset):
    @staticmethod
    def _parse_date(file):
        date_part = file.stem.split('_')[-1]
        date = np.datetime64(date_part)
        return date

    @property
    def _files_template(self):
        return '*oras5_*'

    def _create_grid(self):
        grid_path = sorted(self.path.glob(self._files_template))[0]
        print(f'loading grid from {grid_path}')
        with xr.open_dataset(grid_path, cache=False) as ds:
            lon = ds.coords['longitude'].values
            lat = ds.coords['latitude'].values
            lat, lon = np.meshgrid(lat, lon)
            grid = {'longitude': lon, 'latitude': lat}
        return grid
        
    @staticmethod
    def load_file_vars(filename, variables, times=None):
        # override parent class
        pass