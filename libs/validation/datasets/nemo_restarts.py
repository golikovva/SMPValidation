import os
import re

import numpy as np
import xarray as xr
from datetime import datetime
from pathlib import Path

from libs.validation import Grid
from libs.validation.datasets.base import Dataset
from libs.validation.datasets.nemo import NemoDataset

# Helper functions for mapping restart pieces onto the full domain
# You can place these in a shared module (e.g., libs.validation.datasets.helpers)
def extract_coords(pieces):
    out = {}
    for piece in pieces:
        pdict = {}
        with xr.open_dataset(piece) as f:
            pdict['nav_lat'] = f['nav_lat'].data
            pdict['nav_lon'] = f['nav_lon'].data
        out[piece] = pdict
    return out

def map_piece(piece, domain):
    def extract_common_pair(pair_1, pair_2):
        pair_1 = set([tuple(i) for i in pair_1])
        pair_2= set([tuple(i) for i in pair_2 ])
        return list(pair_1.intersection(pair_2))[0]
        
    mapping = {}
    lat_00 = piece['nav_lat'][0, 0]
    lon_00 = piece['nav_lon'][0, 0]
    lat_11 = piece['nav_lat'][-1, -1]
    lon_11 = piece['nav_lon'][-1, -1]

    dom_lat = domain['nav_lat']
    dom_lon = domain['nav_lon']

    mapping['ind_00'] = extract_common_pair(np.argwhere(dom_lat == lat_00), np.argwhere(dom_lon == lon_00))
    mapping['ind_11'] = extract_common_pair(np.argwhere(dom_lat == lat_11), np.argwhere(dom_lon == lon_11))
    return mapping

def check_coverage(mapping, coords_domain):
    domain = np.ones_like(coords_domain['nav_lat'])
    area = np.sum(domain)
    for piece, coords in mapping.items():
        x0, y0 = coords['ind_00']        
        x1, y1 = coords['ind_11']
        domain[x0:x1 + 1, y0:y1 + 1] = 0
    print(1 - np.sum(domain) / area, 'coverage in percentage  (<100% is ok, because of the land)')
    
def create_mapping(pieces, domain):
    mapping = {}

    pieces = sorted(pieces)

    coords_pieces = extract_coords(pieces)
    with xr.open_dataset(domain) as f:
        coords_domain = {'nav_lat': f['nav_lat'].data,
                         'nav_lon': f['nav_lon'].data
                        }
    for piece, coords in coords_pieces.items():
        #name = f"index_{piece.split('_')[-1][:-3]}"
        mapping[str(piece)] = map_piece(coords, coords_domain)
    check_coverage(mapping, coords_domain)
    
    mapping['size_output'] = coords_domain['nav_lat'].shape
    
    return mapping
    
def map_pieces(mappable, mapping):
    if isinstance(mappable, dict):
        mappable_is_pieces = True
    else:
        mappable_is_pieces = False

    if mappable_is_pieces:
        output_shape = mapping['size_output']
        output = np.zeros(output_shape)
        flag_3d = (len(output_shape)==3)
        for piece, array in mappable.items():
            #name = f"index_{str(piece).split('_')[-1][:-3]}"
            piece_map = mapping[piece]
            if flag_3d:
                output[:, piece_map['ind_00'][0]: piece_map['ind_11'][0] + 1, piece_map['ind_00'][1]: piece_map['ind_11'][1] + 1] = array
            else:
                output[piece_map['ind_00'][0]: piece_map['ind_11'][0] + 1, piece_map['ind_00'][1]: piece_map['ind_11'][1] + 1] = array
    else:
        output = {}
        for name, indexes in mapping.items():
            if name.find('size') > -1:
                continue
            ind_00 = indexes['ind_00']
            ind_11 = indexes['ind_11']
            array = np.copy(mappable[ind_00[0]: ind_11[0] + 1, ind_00[1]: ind_11[1] + 1])
            output[name] = array
    return output

def create_mapping(pieces, domain_cfg):
    """
    Build a mapping from restart pieces to the full domain grid.
    pieces: list of file paths for a single restart date
    domain_cfg: path to the NEMO domain config (.nc) with nav_lat/nav_lon
    """

    mapping = {}
    # compute coords for pieces
    coords_pieces = extract_coords(pieces)
    # load full domain coords
    with xr.open_dataset(domain_cfg) as f:
        coords_domain = { 'nav_lat': f['nav_lat'].data, 'nav_lon': f['nav_lon'].data }
    for piece, coords in coords_pieces.items():
        mapping[str(piece)] = map_piece(coords, coords_domain)
    check_coverage(mapping, coords_domain)
    mapping['size_output'] = coords_domain['nav_lat'].shape
    return mapping


def select_keys_3d(pieces):
    """Identify 3D variables in a restart piece"""
    with xr.open_dataset(pieces[0]) as ds:
        return [var for var in ds.variables if ds[var].dims == ('time_counter', 'nav_lev', 'y', 'x')]


def get_mapped_data(pieces, mapping, keys):
    """Load all 3D restart variables for a given date and map onto full grid"""
    data = {key: {} for key in keys}
    # read raw arrays from each piece
    for piece in pieces:
        with xr.open_dataset(piece) as ds:
            for key in keys:
                data[key][piece] = ds[key].data[0]
    for key in keys:
        data[key] = map_pieces(data[key], mapping)
    return data


import os
import numpy as np
import xarray as xr
from datetime import datetime
from pathlib import Path

from libs.validation import Grid
from libs.validation.datasets.base import Dataset
from libs.validation.datasets.nemo import NemoDataset

class Nemo3DRestartDataset(NemoDataset):
    """
    Dataset for loading 3D NEMO restart fields mapped onto the full domain grid.
    Мэппинг по именам файлов сохраняется для всех дат.
    """
    def __init__(self, path, domain_cfg, name=None, keys=None, *, files_template=None):
        self.domain_cfg = Path(domain_cfg)
        super().__init__(path=path, name=name, files_template=files_template)

        # Построим мэппинг лишь один раз, на примере первой доступной даты
        sample_date = next(iter(self.dates_dict))
        sample_files = self.dates_dict[sample_date]
        # 1) создаём мэппинг по полным путям
        raw_map = self._build_raw_mapping(sample_files, self.domain_cfg)
        # 2) приводим ключи к именам файлов
        self.mapping = {}
        for full_path, coords in raw_map.items():
            fname = Path(full_path).name
            self.mapping[fname] = coords
        # не забываем размер
        self.mapping['size_output'] = raw_map['size_output']

        # ключи 3D-переменных
        self.keys = self._select_keys_3d(sample_files) if keys is None else keys

    @property
    def _default_files_template(self) -> str:
        # ищем все куски рестартов
        return 'run_*/nemo_restart/restart-*.nc'

    @staticmethod
    def _parse_date(file):
        """
        Дата извлекается из имени родительской папки:
        /.../run_2025-05-23T00:00:00+00:00/nemo_restart/...
        """
        run_dir = Path(file).parents[1].name  # 'run_2025-05-23T00:00:00+00:00'
        dt_str = run_dir.replace('run_', '')
        return datetime.fromisoformat(dt_str).date()

    def _create_grid(self):
        with xr.open_dataset(self.domain_cfg) as ds:
            lat = ds['nav_lat'].values
            lon = ds['nav_lon'].values
        return Grid(lat, lon)

    def __getitem__(self, idx):
        """
        - dataset[date] -> dict: { key3d: array3d }

        """
        return self._load_all(idx)

    def _load_all(self, date):
        files = self.dates_dict.get(date)
        if files is None:
            return None
        res_dict = self._get_mapped_data(files, self.mapping, self.keys)
        for k in res_dict:
            res_dict[k] = self._process_field(res_dict[k])
        if len(res_dict) == 1:
            return next(iter(res_dict.values()))
        
        return res_dict

    @staticmethod
    def _build_raw_mapping(pieces, domain_cfg):
        """
        Вспомогательно: строим мэппинг по полным путям, включая 'size_output'
        """
        coords_pieces = extract_coords(pieces)
        with xr.open_dataset(domain_cfg) as f:
            domain_coords = {'nav_lat': f['nav_lat'].data, 'nav_lon': f['nav_lon'].data}
        mp = {}
        for p, coord in coords_pieces.items():
            mp[p] = map_piece(coord, domain_coords)
        check_coverage(mp, domain_coords)
        mp['size_output'] = domain_coords['nav_lat'].shape
        return mp

    @staticmethod
    def _select_keys_3d(pieces):
        with xr.open_dataset(pieces[0]) as ds:
            return [v for v in ds.variables if ds[v].dims == ('time_counter', 'nav_lev', 'y', 'x')]

    @staticmethod
    def _get_mapped_data(pieces, mapping, keys):
        """
        Загружаем по каждому ключу (3D или 2D) массивы и мэпим их в единую сетку.
        Размерность arr определяется автоматически по data.ndim.
        """
        raw = {k: {} for k in keys}
        # 1) Считать все куски
        for p in pieces:
            with xr.open_dataset(p) as ds:
                for k in keys:
                    # ds[k].data[0] → shape либо (lev, y, x), либо (y, x)
                    raw[k][p] = ds[k].data[0]

        out = {}
        base_shape = mapping['size_output']  # (ny, nx)
        for k, piece_dict in raw.items():
            # Определим форму выходного массива
            sample = next(iter(piece_dict.values()))
            if sample.ndim == 3:
                # 3D: (lev, y, x)
                nlev = sample.shape[0]
                arr = np.zeros((nlev, *base_shape))
            else:
                # 2D: (y, x)
                arr = np.zeros(base_shape)

            # Заполняем соответствующими кусками
            for p, data in piece_dict.items():
                fname = Path(p).name
                i0, j0 = mapping[fname]['ind_00']
                i1, j1 = mapping[fname]['ind_11']
                if arr.ndim == 3:
                    arr[:, i0:i1+1, j0:j1+1] = data
                else:
                    arr[i0:i1+1, j0:j1+1] = data
            out[k] = arr
        return out


    def _process_field(self, field):
        """
        Processes the Sea Ice Concentration (SIC) field by applying masking and unit conversion.

        Args:
            field (np.array): Raw SIC data.

        Returns:
            np.array: Processed SIC data.
        """
        field[:, self.src_grid.land_mask()] = np.nan  # Apply land mask
        return field
    
    def _extract_data(self, file, load_fn=None):
        """
        Abstract method to extract data from a file.

        Args:
            file (str): Path to the data file.
            load_fn (callable, optional): Function to load the file. Defaults to xarray's open_dataset.

        Must be implemented by subclasses.
        """
        pass
