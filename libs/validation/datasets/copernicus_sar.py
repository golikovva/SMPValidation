from datetime import datetime

import numpy as np
import xarray as xr
from xarray.coding.times import decode_cf_datetime

from libs.validation.datasets.base import Dataset
from libs.validation.grid import Grid
from libs.validation.grid_utils import lat_lon_to_2d


class CopernicusSarSicDataset(Dataset):
    """Daily Copernicus SAR SIC mosaics, optionally split by acquisition hour.

    Files are named ``sic_YYYYMMDD.nc`` and contain one time layer on a fixed
    latitude/longitude grid. ``dataset[date]`` returns SIC in percent as
    float32, with axes (time, channel, y, x): (1, 1, H, W) in ``daily`` mode
    and (24, 1, H, W) in ``hourly`` mode. Hours are UTC, and unobserved pixels
    remain NaN. Hourly layers partition the mosaic; they do not reconstruct
    observations discarded when the daily mosaic was produced.

    ``coarsen`` is a positive integer or (latitude, longitude) factor pair.
    It builds a smaller grid and averages valid SIC by spherical pixel area
    before interpolation, retaining short edge blocks. The default 1 is unchanged.
    Coarsening requires regular monotonic 1D coordinate axes.

    Spatial interpolation and file indexing use the base Dataset machinery.
    Source daily fields are read in full; hourly coarsening creates only the
    24 reduced output layers. Uncertainty is not read.
    """

    def __init__(
        self, path, dst_grid=None, name=None, *, average_times=None, mode="daily", coarsen=1, 
        files_template=None, interpolation_cache_dir=None, **kwargs
    ):
        if mode not in ("daily", "hourly"):
            raise ValueError("mode must be 'daily' or 'hourly'")
        self.mode = mode
        self.coarsen = self._normalize_coarsen(coarsen)
        super().__init__(
            path, dst_grid=dst_grid, average_times=average_times, name=name,
            files_template=files_template,
            interpolation_cache_dir=interpolation_cache_dir, **kwargs
        )

    @staticmethod
    def _normalize_coarsen(coarsen):
        factors = coarsen if isinstance(coarsen, (tuple, list)) else (coarsen, coarsen)
        if len(factors) != 2 or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
            for value in factors
        ):
            raise ValueError("coarsen must be a positive integer or a pair of positive integers")
        return tuple(int(value) for value in factors)

    @property
    def _default_files_template(self):
        return "**/sic_*.nc"

    @staticmethod
    def _parse_date(file):
        if not file.stem.startswith("sic_"):
            raise ValueError(f"Expected sic_YYYYMMDD filename: {file.name}")
        date_text = file.stem[4:]
        if len(date_text) != 8 or not date_text.isdigit():
            raise ValueError(f"Expected sic_YYYYMMDD filename: {file.name}")
        return datetime.strptime(date_text, "%Y%m%d").date()

    def _create_dates_dict(self):
        dates = super()._create_dates_dict()
        if not dates:
            raise ValueError(f"No Copernicus SAR SIC files found in {self.path}")
        for date, files in dates.items():
            if len(files) != 1:
                raise ValueError(f"Expected one SIC file for {date}, found {len(files)}")
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
        with xr.open_dataset(file, decode_times=False, decode_timedelta=False) as ds:
            self._spatial_dims, self._latitude, self._longitude = self._geometry(ds)
            self._field(ds, "sic")  # Check dimensions without loading SIC.
        lat, lon = self._latitude, self._longitude
        if self.coarsen != (1, 1):
            lat, lon = self._prepare_coarsening()
        lat, lon = lat_lon_to_2d(lat, lon)
        return Grid(lat, lon)

    @staticmethod
    def _axis_bounds(axis, name):
        """Infer cell edges from a regular monotonic one-dimensional axis."""
        values = np.asarray(axis, dtype=np.float64)
        if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
            raise ValueError(f"coarsen requires finite regular 1D {name} coordinates")
        steps = np.diff(values)
        tolerance = 1e-10
        if np.issubdtype(axis.dtype, np.floating):
            tolerance = max(tolerance, 4 * np.finfo(axis.dtype).eps * max(1, np.max(np.abs(values))))
        if not (
            (np.all(steps > 0) or np.all(steps < 0))
            and np.allclose(steps, steps.mean(), rtol=1e-5, atol=tolerance)
        ):
            raise ValueError(f"coarsen requires regular monotonic {name} coordinates")
        if name == "latitude" and np.any(np.abs(values) > 90):
            raise ValueError("latitude coordinates must be between -90 and 90 degrees")

        bounds = np.empty(values.size + 1, dtype=np.float64)
        bounds[1:-1] = (values[:-1] + values[1:]) / 2
        bounds[0] = values[0] - steps[0] / 2
        bounds[-1] = values[-1] + steps[-1] / 2
        if name == "latitude":
            np.clip(bounds, -90, 90, out=bounds)
        return bounds

    def _prepare_coarsening(self):
        """Prepare small axis arrays only; never build the original 2D grid."""
        lat_bounds = self._axis_bounds(self._latitude, "latitude")
        lon_bounds = self._axis_bounds(self._longitude, "longitude")
        shape = (self._latitude.size, self._longitude.size)
        self._coarse_shape = tuple(
            (size - 1) // factor + 1 for size, factor in zip(shape, self.coarsen)
        )
        if min(self._coarse_shape) < 2:
            raise ValueError("coarsen must leave at least two cells along each grid axis")

        self._row_starts = np.arange(0, shape[0], self.coarsen[0])
        self._col_starts = np.arange(0, shape[1], self.coarsen[1])
        row_ends = np.minimum(self._row_starts + self.coarsen[0], shape[0])
        col_ends = np.minimum(self._col_starts + self.coarsen[1], shape[1])
        lat = (lat_bounds[self._row_starts] + lat_bounds[row_ends]) / 2
        lon = (lon_bounds[self._col_starts] + lon_bounds[col_ends]) / 2
        # Spherical cell areas are separable. The common Earth-radius factor
        # cancels when normalizing, including for pole-clipped source cells.
        self._lat_weights = np.abs(np.diff(np.sin(np.deg2rad(lat_bounds))))
        self._lon_weights = np.abs(np.deg2rad(np.diff(lon_bounds)))
        return lat, lon

    def _coarsen_field(self, field, valid=None):
        """Area-weighted means of valid pixels, preserving partial edge blocks."""
        if self.coarsen == (1, 1):
            return field
        finite = np.isfinite(field)
        if valid is not None:
            finite &= valid
        # Reuse one source-sized float64 buffer for numerator and denominator;
        # reduceat includes short edge blocks without padding the source.
        work = np.zeros(field.shape, dtype=np.float64)
        np.copyto(work, field, where=finite)
        work *= self._lat_weights[:, None]
        work *= self._lon_weights[None, :]
        numerator = np.add.reduceat(
            np.add.reduceat(work, self._col_starts, axis=1), self._row_starts, axis=0,
        )
        work[:] = finite
        work *= self._lat_weights[:, None]
        work *= self._lon_weights[None, :]
        denominator = np.add.reduceat(
            np.add.reduceat(work, self._col_starts, axis=1), self._row_starts, axis=0,
        )
        np.divide(numerator, denominator, out=numerator, where=denominator > 0)
        numerator[denominator == 0] = np.nan
        return numerator.astype(np.float32)

    def _create_interpolator(self):
        interpolator = super()._create_interpolator()
        if interpolator is None:
            return None

        def interpolate_float32(field):
            # ESMF returns float64; retain our dtype without changing regridding.
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

    def _validate_grid(self, ds):
        dims, lat, lon = self._geometry(ds)
        if (
            dims != self._spatial_dims
            or not np.array_equal(lat, self._latitude)
            or not np.array_equal(lon, self._longitude)
        ):
            raise ValueError("All SIC files must use the same latitude/longitude grid")

    def _process_field(self, field):
        field = np.asarray(field).astype(np.float32)
        field[~np.isfinite(field) | (field < 0) | (field > 100)] = np.nan
        return field

    @staticmethod
    def _acquisition_times(variable):
        values = variable.values
        if np.issubdtype(values.dtype, np.datetime64):
            return values.astype("datetime64[ns]")
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError("acq_time must contain datetimes or numeric acquisition times")

        # Promote before conversion: float32 epoch arithmetic loses precision.
        values = values.astype(np.float64)
        values[~np.isfinite(values)] = np.nan
        for key in ("_FillValue", "missing_value"):
            if key in variable.attrs:
                for missing in np.asarray(variable.attrs[key]).ravel():
                    values[values == missing] = np.nan

        units = str(variable.attrs.get("units", "")).strip()
        if units == "s":
            # In the product files the Unix epoch is documented in long_name,
            # rather than in a CF-compatible units attribute. Zero is fill.
            values[values == 0] = np.nan
            units = "seconds since 1970-01-01 00:00:00"
        elif "since" not in units:
            raise ValueError("acq_time units must be 's' or CF time units with an epoch")
        return decode_cf_datetime(
            values, units, calendar=variable.attrs.get("calendar", "standard"),
            use_cftime=False,
        )

    def _extract_data(self, file, load_fn=xr.open_dataset):
        # Decode packing/fill values normally, but handle acq_time explicitly:
        # this product's units='s' is not a complete CF time declaration.
        with load_fn(file, decode_times=False, decode_timedelta=False) as ds:
            self._validate_grid(ds)
            sic = self._process_field(self._field(ds, "sic").values)
            if "status_flag" in ds:
                flags = self._field(ds, "status_flag").values
                known = np.isfinite(flags)
                if np.any(flags[known] < 0) or np.any(flags[known] != np.floor(flags[known])):
                    raise ValueError("status_flag must contain nonnegative integer bitmasks")
                land = np.zeros(sic.shape, dtype=bool)
                land[known] = (flags[known].astype(np.int64) & 1) != 0
                sic[land] = np.nan

            if self.mode == "daily":
                return self._coarsen_field(sic)[None, None]

            times = self._acquisition_times(self._field(ds, "acq_time"))
            day = np.datetime64(self._parse_date(file), "ns")
            coarsened = self.coarsen != (1, 1)
            shape = self._coarse_shape if coarsened else sic.shape
            result = np.full((24, 1, *shape), np.nan, dtype=np.float32)
            for hour in range(24):
                start = day + np.timedelta64(hour, "h")
                end = start + np.timedelta64(1, "h")
                valid_time = (times >= start) & (times < end)
                if coarsened:
                    result[hour, 0] = self._coarsen_field(sic, valid_time)
                else:
                    np.copyto(result[hour, 0], sic, where=valid_time)
            return result
