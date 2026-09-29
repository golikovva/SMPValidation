from abc import abstractmethod
from datetime import datetime

import numpy as np
import xarray as xr

from libs.validation.datasets.base import Dataset
from libs.validation.grid import Grid
from libs.validation.grid_utils import lat_lon_to_2d


class OstiaDataset(Dataset):
    """Read one daily OSTIA field on a fixed latitude/longitude grid.

    Subclasses select a variable and convert its CF-decoded, ocean-masked
    values. ``dataset[date]`` returns float32 with axes (time, channel, y, x)
    and shape (1, 1, H, W). Dates without a file return None. The base Dataset
    provides spatial interpolation and its optional weight cache.
    """

    def __init__(
        self, path, dst_grid=None, name=None, *, average_times=None, files_template=None,
        interpolation_cache_dir=None,
    ):
        super().__init__(
            path, dst_grid=dst_grid, average_times=average_times, name=name,
            files_template=files_template,
            interpolation_cache_dir=interpolation_cache_dir,
        )

    @property
    @abstractmethod
    def _variable(self):
        """Name of the NetCDF variable read by this subclass."""
        raise NotImplementedError

    @property
    def _default_files_template(self):
        return "**/*-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB*.nc"

    @staticmethod
    def _parse_date(file):
        date_text = file.name[:8]
        if len(date_text) != 8 or not date_text.isdigit():
            raise ValueError(f"Expected an OSTIA filename starting with YYYYMMDD: {file.name}")
        return datetime.strptime(date_text, "%Y%m%d").date()

    def _create_dates_dict(self):
        dates = super()._create_dates_dict()
        if not dates:
            raise ValueError(f"No OSTIA files found in {self.path}")
        for date, files in dates.items():
            if len(files) != 1:
                raise ValueError(f"Expected one OSTIA file for {date}, found {len(files)}")
        return dates

    @staticmethod
    def _geometry(ds):
        lat, lon = ds["latitude"], ds["longitude"]
        if lat.ndim == lon.ndim == 1 and lat.dims != lon.dims:
            dims = (lat.dims[0], lon.dims[0])
        elif lat.ndim == lon.ndim == 2 and lat.dims == lon.dims:
            dims = lat.dims
        else:
            raise ValueError("latitude/longitude must be 1D axes or matching 2D coordinates")
        return dims, lat.values, lon.values

    def _create_grid(self):
        file = self.dates_dict[min(self.dates_dict)][0]
        with xr.open_dataset(file) as ds:
            self._spatial_dims, self._latitude, self._longitude = self._geometry(ds)
            self._validate_date(ds, file)
            self._validate_variable(self._field(ds, self._variable))
            self._field(ds, "mask")  # Check dimensions without loading the mask.
        lat, lon = lat_lon_to_2d(self._latitude, self._longitude)
        return Grid(lat, lon)

    def _create_interpolator(self):
        interpolator = super()._create_interpolator()
        if interpolator is None:
            return None

        def interpolate_float32(field):
            # Retain the public dtype; the parent's regridding is unchanged.
            return np.asarray(interpolator(field), dtype=np.float32)

        return interpolate_float32

    def _field(self, ds, name):
        field = ds[name]
        if field.dims != ("time", *self._spatial_dims) or ds.sizes.get("time") != 1:
            raise ValueError(
                f"{name} must have one time layer and dimensions "
                f"{('time', *self._spatial_dims)}, got {field.dims} {field.shape}"
            )
        return field.isel(time=0)

    def _validate_date(self, ds, file):
        time = ds["time"]
        if time.dims != ("time",) or time.size != 1:
            raise ValueError(f"{file.name} must have exactly one time layer")
        try:
            actual_date = time.dt.strftime("%Y-%m-%d").item()
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"{file.name} must have a decoded datetime time coordinate") from exc
        expected_date = self._parse_date(file).isoformat()
        if actual_date != expected_date:
            raise ValueError(
                f"{file.name}: time date {actual_date} does not match filename date {expected_date}"
            )

    def _validate_grid(self, ds):
        dims, lat, lon = self._geometry(ds)
        if (
            dims != self._spatial_dims
            or not np.array_equal(lat, self._latitude)
            or not np.array_equal(lon, self._longitude)
        ):
            raise ValueError("All OSTIA files must use the same latitude/longitude grid")

    def _validate_variable(self, variable):
        """Allow subclasses to check metadata before values are loaded."""

    def _extract_data(self, file, load_fn=xr.open_dataset):
        # xarray applies CF packing and fill values exactly once.
        with load_fn(file) as ds:
            self._validate_grid(ds)
            self._validate_date(ds, file)
            variable = self._field(ds, self._variable)
            self._validate_variable(variable)
            field = np.array(variable.values, dtype=np.float64, copy=True)
            mask = self._field(ds, "mask").values
            valid = (
                np.isfinite(mask) & (mask >= 1) & (mask <= 31)
                & (mask == np.floor(mask))
            )
            flags = np.zeros(mask.shape, dtype=np.uint8)
            flags[valid] = mask[valid].astype(np.uint8)
            # Land, lakes and rivers are excluded; the sea-ice bit is retained.
            valid &= (flags & (2 | 4 | 16)) == 0
            field[~valid] = np.nan
            return self._process_field(field)[None, None]


class OstiaSicDataset(OstiaDataset):
    """Daily OSTIA sea ice concentration in percent (0-100)."""

    @property
    def _variable(self):
        return "sea_ice_fraction"

    def _process_field(self, field):
        field = np.array(field, dtype=np.float64, copy=True)
        field[~np.isfinite(field) | (field < 0) | (field > 1)] = np.nan
        return (field * 100).astype(np.float32)


class OstiaSstDataset(OstiaDataset):
    """Daily OSTIA foundation SST in celsius (default) or kelvin.

    ``units`` selects output units. Input analysed_sst must be in kelvin
    after CF decoding; packed scale/offset values are not applied again.
    """

    def __init__(
        self, path, dst_grid=None, name=None, *, average_times=None, units="celsius",
        files_template=None, interpolation_cache_dir=None,
    ):
        if units not in ("celsius", "kelvin"):
            raise ValueError("units must be 'celsius' or 'kelvin'")
        self.units = units
        super().__init__(
            path, dst_grid=dst_grid, name=name, average_times=average_times, files_template=files_template,
            interpolation_cache_dir=interpolation_cache_dir,
        )

    @property
    def _variable(self):
        return "analysed_sst"

    def _validate_variable(self, variable):
        source_units = str(variable.attrs.get("units", "")).strip().lower()
        if source_units not in ("k", "kelvin"):
            raise ValueError(
                "analysed_sst must use K/kelvin after CF decoding; "
                f"got units={variable.attrs.get('units')!r}"
            )

    def _process_field(self, field):
        field = np.array(field, dtype=np.float64, copy=True)
        field[~np.isfinite(field)] = np.nan
        if self.units == "celsius":
            field -= 273.15
        return field.astype(np.float32)
