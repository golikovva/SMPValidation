import os
from abc import abstractmethod
from pathlib import Path
from datetime import datetime
import random
import pickle
import torch
import numpy as np
import pandas as pd
import netCDF4
import wrf
import pygrib
import xarray as xr
from typing import Callable, List, Dict, Tuple, Any
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import PackedSequence

from ..inv_dist_interp import InvDistTree_np, InvDistTree


def atleast_nd(arr, n):
    """
    inplace operator to expand array dims up to n dimensional
    """
    arr.shape = (1,) * (n - arr.ndim) + arr.shape
    return arr


def get_hours_in_period(date, period):
    return ((date.astype(f'datetime64[{period}]') + np.timedelta64(1, period)).astype(f'datetime64[h]') - date.astype(f'datetime64[{period}]')).astype(int)
    
def lon_to_180_range(lon):
    return (lon + 180) % 360 - 180

class NCs2sDataset(Dataset):
    def __init__(self, data_folder, data_variables=None, seq_len=4, time_resolution_h=1, add_coords=False, add_time_encoding=False):
        super().__init__()
        self.path = Path(data_folder)
        self.dates_dict = self._create_dates_dict()
        self.data_variables = data_variables

        self.constant_vars = {}
        self.src_grid = self._create_grid()
        self.src_grid['longitude'] = lon_to_180_range(self.src_grid['longitude'])   # todo check for tupost/costylnost
        self.seq_len = seq_len
        self.file_len = 24
        self.time_res_h = time_resolution_h
        self.add_coords = add_coords
        self.add_time_encoding = add_time_encoding

    def _create_dates_dict(self):
        result = {}
        files = sorted(self.path.glob(self._files_template))
        for file in files:
            try:
                date = self._parse_date(file).astype(f'datetime64[{self._file_len}]')
            except ValueError:
                print(f'skipping {file}')
                continue
            if date in result:
                result[date].append(file)
            else:
                result[date] = [file]
        print(f'parsed {len(result)} dates')
        return result
    
    @abstractmethod
    def _create_grid(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _parse_date(file):
        raise NotImplementedError

    @property
    @abstractmethod
    def _files_template(self):
        raise NotImplementedError

    @staticmethod
    def load_file_vars(filename, variables, times=None):
        times = wrf.ALL_TIMES if times is None else times
        npy = []
        with netCDF4.Dataset(filename, 'r') as ncf:
            for i, variable in enumerate(variables):
                var = wrf.getvar(ncf, variable, times, meta=False, squeeze=False)
                var = atleast_nd(var, 5)[:, 0]
                npy.append(var)
        npy = np.concatenate(npy, 0)
        return np.transpose(npy, (1, 0, 2, 3))

    def __len__(self):
        l = len(self.dates_dict) * self.file_len - self.seq_len + 1
        return l if l > 0 else 0

    def get_data_by_id(self, date, length=None):
        needed_len = self.seq_len if length is None else length
        npys = []
        hour = (date.astype('datetime64[h]') - date.astype(f'datetime64[{self._file_len}]')).astype(int)
        file = date.astype(f'datetime64[{self._file_len}]')
        while needed_len > 0:
            times = np.arange(0, get_hours_in_period(date, self._file_len))[hour:hour + needed_len*self.time_res_h:self.time_res_h]
            try:
                file_vars = self.load_file_vars(str(self.dates_dict[file][0]), self.data_variables, times)
            except IndexError:
                print(date)
                print("No data for this dates")
                return None
            hour = 0
            file = file + np.timedelta64(1, self._file_len)
            npys.append(file_vars)
            needed_len -= len(file_vars)
        return np.concatenate(npys) 

    def __getitem__(self, date, length=None, add_coords=None, add_time_encoding=None):
        data = [self.get_data_by_id(date, length)]
        add_coords = self.add_coords if add_coords is None else add_coords
        add_time_encoding = self.add_time_encoding if add_time_encoding is None else add_time_encoding
        if add_coords:
            data.append(np.broadcast_to(np.stack([self.src_grid['latitude'], self.src_grid['longitude']]), [self.seq_len, 2, *data[0].shape[-2:]]))
        if self.add_time_encoding:
            day_encoded, hour_encoded = self.get_day_hour_encoding(date)
            data.extend([np.expand_dims(day_encoded, 1), np.expand_dims(hour_encoded, 1)])
        data = np.concatenate(data, axis=1)
        return data
    
    def get_day_hour_encoding(self, date, frequency=1):
        date_seq = np.arange(date, date + np.timedelta64(self.seq_len*self.time_res_h, 'h'), np.timedelta64(self.time_res_h, 'h'))
        day = (date_seq.astype('datetime64[D]') - date_seq.astype('datetime64[Y]')).astype(int) / 365
        hour = (date_seq.astype('datetime64[h]') - date_seq.astype('datetime64[D]')).astype(int) / 24
        d1 = abs(abs(0.5 - day) - 0.5) + 0.05
        d2 = abs(abs(0.25 - day) - 0.5) + 0.05
        h1 = abs(abs(0.5 - hour) - 0.5) + 0.05
        h2 = abs(abs(0.25 - hour) - 0.5) + 0.05
        x, y = np.arange(0, self.src_grid['latitude'].shape[1], 1), np.arange(0, self.src_grid['latitude'].shape[0], 1)
        xxyy = np.meshgrid(x, y)
        day_encoded = (np.sin(frequency * np.einsum('i,jk->ijk', d1, xxyy[0]))
                       + np.sin(frequency * np.einsum('i,jk->ijk', d2, xxyy[1]))) / 2
        hour_encoded = (np.cos(frequency * np.einsum('i,jk->ijk', h1, xxyy[0]))
                        + np.cos(frequency * np.einsum('i,jk->ijk', h2, xxyy[1]))) / 2
        return day_encoded, hour_encoded


class GFSDataset(NCs2sDataset):
    def _create_grid(self):
        grid_path = sorted(self.path.glob(self._files_template))[0]
        print(f'loading grid from {grid_path}')
        with pygrib.open(grid_path) as ds:
            # Read the first message (assuming grid is consistent across messages)
            grb = ds.read(1)[0]
            # Extract latitude/longitude arrays (2D grids)
            lat, lon = grb.latlons()

        # Create and return a Grid object
        grid = {'longitude': lon, 'latitude': lat}
        return grid

    def load_file_vars(self, filename, variables, times):
        out = []
        print(filename)
        with pygrib.open(filename) as ds:
            for i in variables:
                try:
                    layer = [k for k in ds.select(name=i)]
                except ValueError:
                    print('Value Error')
                # check if param unique in grib
                if len(layer) != 1:
                    print(f'file has {len(layer)} params, must be 1')
                layer = layer[0]
                geophysical_data, lats, lons = layer.data()
                out.append(geophysical_data)
        return np.stack(out)[None]

    @staticmethod
    def _parse_date(file):
        # Extract the date part from the filename and parse it
        date_part = file.stem.split('.')[-2]
        date = datetime.strptime(date_part, '%Y%m%d%H')  # Parse as 'YYYYMMDDHH'
        date = np.datetime64(date)
        return date
    
    @property
    def _files_template(self):
        return 'gfs.*.f*.grib2'
    
    @property
    def _file_len(self):
        return '6h'


class WRFs2sDataset(NCs2sDataset):
    @staticmethod
    @abstractmethod
    def _parse_date(file):
        date_part = file.name.split('_')[2]
        date = np.datetime64(date_part)
        return date

    @property
    def _files_template(self):
        return '*wrfout*'

    def _create_grid(self):
        grid_path = sorted(self.path.glob(self._files_template))[0]
        print(f'loading grid from {grid_path}')
        with xr.open_dataset(grid_path, cache=False) as ds:
            lon = ds.coords['XLONG'].values[0]
            lat = ds.coords['XLAT'].values[0]
            grid = {'longitude': lon, 'latitude': lat}
        return grid
    
    @property
    def _file_len(self):
        return 'D'

class ERAs2sDataset(NCs2sDataset):
    @staticmethod
    @abstractmethod
    def _parse_date(file):
        date_part = file.name.split('_')[-1]
        date = np.datetime64(date_part)
        return date

    @property
    def _files_template(self):
        return '*era*'

    def _create_grid(self):
        grid_path = sorted(self.path.glob(self._files_template))[0]
        print(f'loading grid from {grid_path}')
        with xr.open_dataset(grid_path, cache=False) as ds:
            lat_vec = ds.coords['latitude'].values
            lon_vec = ds.coords['longitude'].values

            lat = np.tile(lat_vec, (lon_vec.size, 1)).T
            lon = np.tile(lon_vec, (lat_vec.size, 1))

            grid = {'longitude': lon, 'latitude': lat}
        return grid
    
    @property
    def _file_len(self):
        return 'D'
        
class ERAMonthlyDataset(NCs2sDataset):
    @staticmethod
    def _parse_date(file):
        """
        Extract year and month from monthly file name (e.g., ERA5_2023-10.nc).
        """
        date_part = file.name.split('_')[-1].split('.')[0]  # Extract "2023-10"
        date = np.datetime64(date_part + '-01')  # Convert to first day of the month
        return date

    def arange_month(self, date):
        month = date.astype('datetime64[M]')
        res = np.arange(
            month.astype('datetime64[D]'),
            (month + np.timedelta64(1, 'M')).astype('datetime64[D]'),
            np.timedelta64(1, 'D')
        )
        return res
    
    @property
    def _files_template(self):
        return 'era*.nc'

    @property
    def _file_len(self):
        return 'M'
        
    def _create_grid(self):
        """
        Create grid from the first file.
        """
        grid_path = sorted(self.path.glob(self._files_template))[0]
        print(f'Loading grid from {grid_path}')
        with xr.open_dataset(grid_path) as ds:
            lat_vec = ds.coords['latitude'].values
            lon_vec = ds.coords['longitude'].values

            lat = np.tile(lat_vec, (lon_vec.size, 1)).T
            lon = np.tile(lon_vec, (lat_vec.size, 1))

            grid = {'longitude': lon, 'latitude': lat}
        return grid
        
class ERAMonthlyDataset_nh(ERAMonthlyDataset):
    def __init__(self, data_folder, data_variables=None, seq_len=4, time_res_h=6):
        super().__init__(data_folder, data_variables, seq_len)
        self.time_res_h = time_res_h
        
    def get_data_by_id(self, date, length=None):
        needed_len = self.seq_len if length is None else length
        npys = []
        hour = (date.astype('datetime64[h]') - date.astype(f'datetime64[{self._file_len}]')).astype(int)
        file = date.astype(f'datetime64[{self._file_len}]')
        while needed_len > 0:
            times = np.arange(0, get_hours_in_period(date, self._file_len))[hour:hour + needed_len*self.time_res_h:self.time_res_h]
            try:
                file_vars = self.load_file_vars(str(self.dates_dict[file][0]), self.data_variables, times)
            except IndexError:
                print(date)
                print("No data for this dates")
                return None
            hour = 0
            file = file + np.timedelta64(1, self._file_len)
            npys.append(file_vars)
            needed_len -= len(file_vars)
        return np.concatenate(npys) 


class ScatterNoneDataset(Dataset):
    def __getitem__(self, index, *args, **kwargs):
        return None


class StationsNoneDataset(Dataset):
    def __getitem__(self, index, *args, **kwargs):
        return None


class StationsDataset(Dataset):
    def __init__(self, stations_folder, data_variables=None, seq_len=4):
        self.path = Path(stations_folder)
        self.station_files = sorted(self.path.glob(self._files_template))
        self.data_variables = data_variables
        self.seq_len = seq_len

        self.coords = None
        self.dates = None
        self.stations = self.load_stations(self.station_files)
        self.dates_dict = self._create_dates_dict()
        self.src_grid = self._create_grid()

    def _create_grid(self):
        grid = {'longitude': self.coords[:, 0], 'latitude': self.coords[:, 1]}
        return grid

    def _create_dates_dict(self):
        np_ids = np.arange(0, self.dates.shape[0])
        date_ids = pd.Series(np_ids, index=self.dates)
        return date_ids

    def load_stations(self, station_files):
        names = []
        coords = []
        stations = []
        for file in station_files:
            with open(file, 'rb') as f:
                measurements = pickle.load(f)
                names.append(measurements['Name'])
                coords.append(measurements['Coords'])
                stations.append(measurements['Station'])
        stations = np.swapaxes(np.array(stations), 0, 1)  # size (40369, 46, 4)
        self.dates = stations[:, 0, 0].astype('datetime64[s]').astype('datetime64[h]')
        self.stations = stations[:, :, 1:]
        self.coords = np.array(coords)
        self.coords[:, [0, 1]] = self.coords[:, [1, 0]]
        return stations

    def __getitem__(self, date_index, length=None):
        length = self.seq_len if length is None else length
        start = date_index
        stop = date_index + np.timedelta64(length-1, 'h')
        ids = self.dates_dict[start:stop]
        return self.stations[ids]

    @property
    def _files_template(self):
        return '*.pkl'


class ScatterDataset(Dataset):
    def __init__(self, file_paths, sequence_len=24):
        """
        Args:
            file_paths (list of str): List of file paths to xarray datasets.
            sequence_len (int): Length of the desired scatter time sequence in hours.
        """
        if os.isdir(file_paths):
            self.file_paths = sorted(self.path.glob(self._files_template))
        else:
            self.file_paths = file_paths
        self.sequence_len = sequence_len
        self.datasets = self._load_datasets()
        self.time_index = self._build_time_index()
        self.src_grid = self._create_grid()  # Create the grid

    def _load_datasets(self):
        """Load all xarray datasets from the file paths."""
        datasets = []
        for file_path in self.file_paths:
            ds = xr.open_dataset(file_path)
            datasets.append(ds)
        return datasets

    def _build_time_index(self):
        """Build a unified time index across all xarray datasets."""
        time_index = []
        for ds in self.datasets:
            time_index.extend(ds.time.values)
        return np.sort(np.unique(time_index))

    def _create_grid(self):
        """
        Create a grid from the first dataset.

        Returns:
            dict: A dictionary containing the longitude and latitude grid.
        """
        # Use the first dataset to extract latitude and longitude
        ds = self.datasets[0]
        lat_vec = ds.coords['latitude'].values
        lon_vec = ds.coords['longitude'].values

        # Create a grid using numpy's tile function
        lat = np.tile(lat_vec, (lon_vec.size, 1)).T
        lon = np.tile(lon_vec, (lat_vec.size, 1))

        # Return the grid as a dictionary
        grid = {'longitude': lon, 'latitude': lat}
        return grid

    def __len__(self):
        """Return the total number of possible sequences."""
        return len(self.time_index) - self.sequence_len

    def _get_time_index(self, idx):
        """Convert idx to a valid time index if it's a np.datetime64 object."""
        if isinstance(idx, np.datetime64):
            # Find the index of the closest time in the time index
            time_diff = np.abs(self.time_index - idx)
            idx = np.argmin(time_diff)
        return idx
    
    @property
    def _files_template(self):
        return '*ascat*.nc'
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int or np.datetime64): Index or timestamp of the starting time.

        Returns:
            dict: A dictionary containing the scatterometer data for the sequence.
        """
        # Convert idx to a valid time index if it's a np.datetime64 object
        idx = self._get_time_index(idx)
        start_time = self.time_index[idx]
        end_time = start_time + np.timedelta64(self.sequence_len-1, 'h')

        # Initialize lists to store the data for the sequence
        measurement_time_seq = []
        northward_wind_seq = []
        eastward_wind_seq = []

        for ds in self.datasets:
            # Select the time slice within the sequence
            time_slice = ds.sel(time=slice(start_time, end_time))

            if len(time_slice.time) > 0:
                measurement_time_seq.append(time_slice.measurement_time.values)
                northward_wind_seq.append(time_slice.northward_wind.values)
                eastward_wind_seq.append(time_slice.eastward_wind.values)

        # Stack the data along the time dimension
        if measurement_time_seq:  # Check if the list is not empty
            measurement_time_seq = np.concatenate(measurement_time_seq, axis=0)
            northward_wind_seq = np.concatenate(northward_wind_seq, axis=0)
            eastward_wind_seq = np.concatenate(eastward_wind_seq, axis=0)
        else:
            # If no data is found, return empty arrays
            measurement_time_seq = np.array([])
            northward_wind_seq = np.array([])
            eastward_wind_seq = np.array([])

        # Return the data as a dictionary
        return {
            'measurement_time': measurement_time_seq.astype('datetime64[ns]').astype('datetime64[s]').astype('float64'),
            'northward_wind': northward_wind_seq,
            'eastward_wind': eastward_wind_seq
        }


class IFSs2sDataset(NCs2sDataset):
    @staticmethod
    def _parse_date(file):
        date_str = file.name.split('.')[-2]
        if len(date_str) != 8 or not date_str.isdigit():
            raise ValueError(f"Invalid date format: {file.name}")
        return np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
        
    def _create_dates_dict(self):
        """Create nested dict: {date: {var: file_path}}"""
        result = {}
        files = sorted(self.path.glob(self._files_template))
        for file in files:
            date = self._parse_date(file)
            var = self._parse_variable(file)
            if date not in result:
                result[date] = {}
            result[date][var] = file
        return result

    @staticmethod
    def _parse_variable(file):
        """Extract variable name from filename (second component after split)"""
        return file.name.split('.')[1]

    @property
    def _files_template(self):
        return '*ec.oper*.*.nc'  # Adjust based on actual file pattern

    def _create_grid(self):
        grid_path = next(self.path.glob(self._files_template))
        print(f'Loading IFS grid from {grid_path}')
        with xr.open_dataset(grid_path) as ds:
            if 'latitude' in ds.variables and 'longitude' in ds.variables:
                lat = ds['latitude'].values
                lon = ds['longitude'].values
                if len(lat.shape) == 1:  # 1D vectors
                    lon, lat = np.meshgrid(lon, lat)
                grid = {'longitude': lon, 'latitude': lat}
                self.constant_vars.update(grid)
            else:
                raise ValueError("Latitude/Longitude variables not found")
        return grid

    def get_data_by_id(self, start_date, length=None):
        seq_len = self.seq_len if length is None else length
        npys = []
        current_time = pd.Timestamp(start_date)
        for _ in range(seq_len):
            # Convert to numpy datetime64 and floor to nearest day
            current_day = current_time.to_datetime64().astype('datetime64[D]')
            
            # Calculate initial time parameters
            if current_time.hour < 6:
                file_date = current_day - np.timedelta64(1, 'D')
                init_hour = 12  # Previous day's 12Z initialization
            elif current_time.hour < 18:
                file_date = current_day
                init_hour = 0   # Same day's 00Z initialization
            else:
                file_date = current_day
                init_hour = 12  # Same day's 12Z initialization

            # Convert file_date to proper datetime64[D] for dict lookup
            file_date_d = file_date.astype('datetime64[D]')
            
            # Check date availability
            if file_date_d not in self.dates_dict:
                return None

            # Calculate forecast hour
            init_time = pd.Timestamp(file_date_d) + pd.DateOffset(hours=init_hour)
            forecast_hour = (current_time - init_time).total_seconds() // 3600

            # Validate forecast hour
            if forecast_hour not in [6, 12]:
                return None

            # Get files for all variables
            date_files = self.dates_dict[file_date_d]
            missing_vars = [var for var in self.data_variables if var not in date_files and var not in self.constant_vars] 
            if missing_vars:
                return None

            # Load and stack variables
            var_data = []
            for var in self.data_variables:
                if var in self.constant_vars:
                    var_data.append(self.constant_vars[var])
                else:
                    with xr.open_dataset(date_files[var]) as ds:
                        # Select initialization time (0 or 1) and forecast hour (0=6h, 1=12h)
                        init_idx = 0 if init_hour == 0 else 1
                        fhour_idx = 0 if forecast_hour == 6 else 1
                        var_data.append(ds[var][init_idx, fhour_idx].values)
            
            npys.append(np.stack(var_data, axis=0))  # (vars, H, W)
            current_time += pd.DateOffset(hours=6)
        
        return np.stack(npys, axis=0)  # (time, vars, H, W)


class StackDataset(Dataset):
    def __init__(self, *datasets):
        self._length = len(datasets[0])
        self.datasets = datasets

    def __getitem__(self, index):
        return tuple(dataset[index] for dataset in self.datasets)

    def __len__(self):
        return self._length
    
class StackVSDataset(StackDataset):
    def __init__(self, *datasets, max_sl=None):
        super().__init__(*datasets)
        self.max_sl = self.datasets[0].seq_len if max_sl is None else max_sl

    def __getitem__(self, index, cur_sl=None):
        cur_sl = random.randint(1, self.max_sl) if cur_sl is None else cur_sl
        return tuple(dataset.__getitem__(index, cur_sl) for dataset in self.datasets)

def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.

    e.g. MNISTWithIndices = dataset_with_indices(MNIST)
         dataset = MNISTWithIndices('~/datasets/mnist')
    """

    def __getitem__(self, index):
        data = cls.__getitem__(self, index)
        return *data, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


class Sampler:
    def __init__(self, days, shuffle=False):
        self.days = days
        self.shuffle = shuffle

    def __len__(self):
        return len(self.days)

    def __iter__(self):
        ids = np.arange(len(self.days))
        if self.shuffle:
            np.random.shuffle(ids)
        for i in ids:
            yield self.days[i]


def variable_len_collate(batch):
    """
    Custom collate function for handling batches with None values and variable-length sequences.

    Args:
        batch (list): A list of data samples.
    elif isinstance(elem, torch.Tensor) and any(len(b) != len(elem) for b in batch):
    Returns:
        Collated batch data, which can be None, a list of datetime64 objects, a packed sequence of tensors,
        or a default collated batch depending on the input batch type.
    """
    elem = batch[0]
    if isinstance(elem, type(None)):
        return None
    elif isinstance(elem, np.datetime64):
        return batch
    elif isinstance(elem, np.ndarray):
        return variable_len_collate([torch.from_numpy(item) for item in batch])
    elif isinstance(elem, torch.Tensor):
        return pack_sequence(batch, enforce_sorted=False)
    elif isinstance(elem, tuple):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.
        return [variable_len_collate(samples) for samples in transposed]  # Backwards compatibility.
    else:
        return default_collate(batch)


def transform_packed_sequence_multiple(
        packed: PackedSequence,
        transforms: List[Tuple[Callable[[torch.Tensor, Any], torch.Tensor], Tuple[Any, ...], Dict[str, Any]]]
    ) -> PackedSequence:
    """
    Последовательно применяет несколько функций преобразования к данным внутри PackedSequence.
    
    Args:
        packed (PackedSequence): Исходный PackedSequence.
        transforms (List[Tuple[Callable, Tuple, Dict]]): Список преобразований, где:
            - первый элемент: функция преобразования,
            - второй элемент: кортеж позиционных аргументов для функции,
            - третий элемент: словарь именованных аргументов для функции.
    
    Returns:
        PackedSequence: Новый PackedSequence с преобразованными данными.
    """
    data = packed.data  # Исходные данные PackedSequence

    # Применяем каждую функцию последовательно
    for transform_fn, args, kwargs in transforms:
        data = transform_fn(data, *args, **kwargs)

    # Создаем новый PackedSequence с преобразованными данными
    return PackedSequence(data, packed.batch_sizes, packed.sorted_indices, packed.unsorted_indices)



def get_novaya_zemlya_mask(fill_value=torch.nan, return_vertices=False):
    nz_vertices = np.array([
        [120, 120],
        [70, 130],
        [10, 190],
        [0, 240],
        [40, 280],
        [160, 160],
    ])
    nz_polygon_array = torch.from_numpy(create_polygon([210, 280], nz_vertices))
    nz_polygon_array = torch.where(nz_polygon_array == 0, fill_value, 1)
    if not return_vertices:
        return nz_polygon_array
    return nz_polygon_array, nz_vertices


def check(p1, p2, base_array):
    """
    Uses the line defined by p1 and p2 to check array of
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """
    idxs = np.indices(base_array.shape)  # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
    sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign


def create_polygon(shape, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero"""
    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros

    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k - 1], vertices[k], base_array)], axis=0)

    # Set all values inside polygon to one
    base_array[fill] = 1

    return base_array

    
def dataset_with_interpolator(cls):
    """
    Modifies the given Dataset class to return a interpolated data
    instead of just data.

    e.g. MNISTWithIndices = dataset_with_indices(MNIST)
         dataset = MNISTWithIndices('~/datasets/mnist')
    """
    def _create_interpolator(self):
        """
        Creates an interpolator if a destination grid is specified.

        Returns:
            Interpolator or None: Interpolator object if dst_grid is provided, else None.
        """
        if self.dst_grid is None:
            return None
        src_nodes = np.stack([self.src_grid['longitude'].flatten(), self.src_grid['latitude'].flatten()]).T
        dst_nodes = np.stack([self.dst_grid['longitude'].flatten(), self.dst_grid['latitude'].flatten()]).T
        print('initializing interpolator')
        interpolator = InvDistTree_np(src_nodes, dst_nodes)  # Initialize interpolator
        return interpolator
    
    def __init__(self, *args, **kwargs):
        self.dst_grid = kwargs.pop('dst_grid', None)
        super(cls, self).__init__(*args, **kwargs)
        self.interpolator = self._create_interpolator()

    def __getitem__(self, index, *args, **kwargs):
        data = super(cls, self).__getitem__(index, *args, **kwargs)
        if self.interpolator is not None:
            s = data.shape
            data = self.interpolator(data.reshape(*s[:-2], -1)).reshape(*s[:-2], *self.dst_grid['longitude'].shape)
        return data

    return type(
        f"{cls.__name__}WithInterpolator",
        (cls,),
        {
            '_create_interpolator': _create_interpolator,
            '__init__': __init__,
            '__getitem__': __getitem__,
        }
    )