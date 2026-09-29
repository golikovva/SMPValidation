import torch
import numpy as np
import geopandas as gpd
from collections.abc import Mapping
from shapely import contains_xy
from datetime import timedelta
from typing import Union
from libs.validation.datasets.base import Dataset
from libs.validation.inv_dist_interp import InvDistTree_np


class DelayedDataset:
    """
    A transparent wrapper around any object implementing the Dataset protocol.
    When accessed with a date key, it returns data for (date - delay).

    All other attributes/methods (grid, dates_dict, name, __len__, etc.)
    are delegated to the underlying dataset.
    """

    def __init__(self, base_ds: Dataset, delay: Union[int, timedelta]):
        """
        Parameters
        ----------
        base_ds : Dataset
            The original dataset instance to wrap.
        delay : int | timedelta
            If int, interpreted as the number of days to shift backward.
        """
        if isinstance(delay, int):
            delay = timedelta(days=delay)

        self._base_ds: Dataset = base_ds
        self._delay: timedelta = delay

        # Helpful suffix for reports/validators so delayed datasets are distinguishable
        suffix = f"t-{delay.days}d"
        self.name: str = f"{getattr(base_ds, 'name', base_ds.__class__.__name__)}_{suffix}"

    def __getattr__(self, item: str):
        """
        Delegate all attribute access (except the ones we redefine)
        to the underlying dataset.
        """
        return getattr(self._base_ds, item)

    def __getitem__(self, date):
        """
        Return data for (date - self._delay).  If the underlying dataset
        returns None, we propagate that None unchanged.
        """
        return self._base_ds[date - self._delay]

    def __len__(self) -> int:
        """Delegate len(...) to the base dataset."""
        return len(self._base_ds)
    

class DynamicsDataset(DelayedDataset):
    """
    A DelayedDataset that computes the difference between the base dataset
    and the delayed dataset.
    """ 
    def __init__(self, base_ds, delay):
        super().__init__(base_ds, delay)
        self.name = f"{self.name}_dynamics"
    def __getitem__(self, date):
        return self._base_ds[date] - super().__getitem__(date)


def with_delay(dataset: Dataset, delay: Union[int, timedelta]) -> DelayedDataset:
    """Utility function for concise wrapping."""
    return DelayedDataset(dataset, delay)

def with_dynamics(dataset: Dataset, delay: Union[int, timedelta]) -> DynamicsDataset:
    """Utility function for concise wrapping."""
    return DynamicsDataset(dataset, delay)

class SeaIceExtentDataset:
    def __init__(self, concentration_dataset, threshold=15):
        self.concentration_dataset = concentration_dataset
        self.threshold = threshold

    def __getitem__(self, idx):
        # Retrieve the sea ice concentration data
        sic = self.concentration_dataset[idx]
        # Convert to sea ice extent (binary mask based on threshold)
        sie = np.where(sic >= self.threshold, 100.0, 0.0)
        return sie
    
def sic_to_sie(sic_dataset_class, threshold=15):
    """
    Decorator that transforms a Sea Ice Concentration (SIC) dataset class 
    into a Sea Ice Extent (SIE) dataset class by applying a 15% threshold.
    
    Args:
        sic_dataset_class: The SIC dataset class to be decorated.
        
    Returns:
        A new class that converts SIC values to binary SIE (0 or 1) based on the 15% threshold.
    """
    
    class SIEDataset(sic_dataset_class):
        """
        Sea Ice Extent (SIE) dataset class that applies a 15% threshold to SIC data.
        """
        
        def _process_field(self, field):
            """
            Processes the SIC field by applying a 15% threshold to convert to SIE.
            
            Args:
                field (np.array): Raw SIC data (0-100 scale).
                
            Returns:
                np.array: Binary SIE data (0 or 1) where 1 indicates ice extent.
            """
            # First apply the original SIC processing (like land masking)
            field = super()._process_field(field)
            
            # Prepare output: start with NaNs everywhere
            sie = np.full_like(field, np.nan, dtype=np.float32)

            # Mask of valid values
            valid = ~np.isnan(field)

            # Apply threshold only to valid pixels
            sie[valid] = (field[valid] >= threshold).astype(np.float32)
            
            return sie
        
        def __repr__(self):
            """
            Returns a string representation of the SIE dataset.
            """
            return f"<{self.__class__.__name__} (15% threshold) wrapping {sic_dataset_class.__name__}>"
    
    # Copy the original class name for better identification
    SIEDataset.__name__ = f"{sic_dataset_class.__name__}_as_SIE"
    SIEDataset.__qualname__ = f"{sic_dataset_class.__qualname__}_as_SIE"
    
    return SIEDataset


def label_seas_on_grid(lats: np.ndarray,
                       lons: np.ndarray,
                       seas_gdf: gpd.GeoDataFrame,
                       label_field: str = None,
                       drop_unused: bool = False,
                       ) -> tuple[np.ndarray, dict]:
    """
    Produce a 2D integer mask for *any* 2D lats/lons arrays, by
    looping over each sea polygon and burning in its index.

    Parameters
    ----------
    lats, lons : np.ndarray, shape (M, N)
        2D arrays of the latitude and longitude of each grid point.
    seas_gdf : geopandas.GeoDataFrame
        The SeaVoX polygons (one per row), must have a valid `geometry`.
    label_field : str, optional
        If provided, use this integer column as the mask label.
        Otherwise polygons are labeled 1,2,… in GeoDataFrame order.

    Returns
    -------
    mask : np.ndarray[int], shape (M, N)
        Integer mask, 0 = land (no sea), >0 = sea‐polygon label.
    lookup : dict[int → dict]
        Maps each nonzero label to a dict with keys:
          - 'gdf_index': original row index in seas_gdf
          - 'properties': all the row’s attributes (as dict)
    """
    # Prepare output
    M, N = lats.shape
    mask = np.zeros((M, N), dtype=np.int32)
    lookup = {}

    # Loop polygons
    next_label = 1
    for idx, row in seas_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            print('geom is empty')
            continue

        # choose the integer label
        if label_field is not None:
            label = int(row[label_field])
            if label == 0:
                raise ValueError(f"label_field '{label_field}' must be nonzero, got 0 at row {idx}")
        else:
            label = next_label
            next_label += 1

        # quick bbox pre‐filter: only test points whose lon/lat lie inside the poly bbox
        minx, miny, maxx, maxy = geom.bounds
        bbox_mask = ((lons >= minx) & (lons <= maxx)
                  & (lats >= miny) & (lats <= maxy))
        if not np.any(bbox_mask):
            # no grid‐points in this polygon’s bbox → skip
            continue

        # vectorized test: True where (lon,lat) is inside geom
        hits = contains_xy(geom, lons, lats)
        # burn into the label mask
        mask[hits] = label

        # record lookup
        lookup[label] = {
            'gdf_index': idx,
            'properties': row.drop('geometry').to_dict()
        }

    return mask, lookup

def make_sea_mask_from_shapefile(lats, lons, shapefile_path):
    gdf = gpd.read_file(shapefile_path)
    return label_seas_on_grid(lats, lons, gdf)

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


def surface_dataset(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.

    e.g. MNISTWithIndices = dataset_with_indices(MNIST)
         dataset = MNISTWithIndices('~/datasets/mnist')
    """

    def __getitem__(self, index):
        data = cls.__getitem__(self, index)[[0,]]
        return data

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

class SurfaceDataset:
    """
    A transparent wrapper around any object implementing the Dataset protocol.
    When accessed with a date key, it returns data for surface layer.

    All other attributes/methods (grid, dates_dict, name, __len__, etc.)
    are delegated to the underlying dataset.
    """

    def __init__(self, base_ds: Dataset):
        """
        Parameters
        ----------
        base_ds : Dataset
            The original dataset instance to wrap.
        """

        self._base_ds: Dataset = base_ds
        # Helpful suffix for reports/validators so delayed datasets are distinguishable
        suffix = f"surface"
        self.name: str = f"{getattr(base_ds, 'name', base_ds.__class__.__name__)}_{suffix}"

    def __getattr__(self, item: str):
        """
        Delegate all attribute access (except the ones we redefine)
        to the underlying dataset.
        """
        return getattr(self._base_ds, item)

    def __getitem__(self, date):
        """
        Return data for (date - self._delay).  If the underlying dataset
        returns None, we propagate that None unchanged.
        """
        return self._base_ds[date][[0]]

    def __len__(self) -> int:
        """Delegate len(...) to the base dataset."""
        return len(self._base_ds)
    
def surfaced(dataset: Dataset) -> DelayedDataset:
    """Utility function for concise wrapping."""
    return SurfaceDataset(dataset)


class SIVDataset:
    def __init__(self, sic_dataset, thick_dataset, mesh_mask):
        self.name = f"{sic_dataset.name}->siv"
        self.grid_area = mesh_mask.e1t * mesh_mask.e2t
        self.grid_area = np.where(self.grid_area<1e10, self.grid_area, np.nan)
        self.sic_dataset = sic_dataset
        self.thick_dataset = thick_dataset
        self.grid = sic_dataset.grid

    def __getitem__(self, idx):
        sic = self.sic_dataset[idx]
        if np.nanmax(sic) > 1:
            sic = sic / 100.0
        thick = self.thick_dataset[idx]
        # Calculate sea ice volume
        siv = sic * thick * self.grid_area
        return siv
    
class SIADataset:
    def __init__(self, sic_dataset, mesh_mask):
        self.name = f"{sic_dataset.name}->sia"
        self.grid_area = mesh_mask.e1t * mesh_mask.e2t
        self.grid_area = np.where(self.grid_area<1e10, self.grid_area, np.nan)
        self.sic_dataset = sic_dataset
        self.grid = sic_dataset.grid
        
    def __getitem__(self, idx):
        sic = self.sic_dataset[idx]
        if np.nanmax(sic) > 1:
            sic = sic / 100.0
        # Calculate sea ice volume
        sia = sic * self.grid_area
        return sia


class InterpolatedOverBuoysDataset:
    """Sample a grid at buoy positions while retaining their point identity.

    New BuoyBatch inputs use their fixed (buoy, time, coordinate) layout.
    Optional model var_names and units describe the model channels;
    observation channel names are only retained as an explicit sampling hint.
    """

    def __init__(
        self, base_ds, buoy_ds, base_time_axis=0, buoy_time_axis=0,
        static=False, lat_first=True, *, var_names=None, units=None,
    ):
        self.base_ds = base_ds
        self.buoy_ds = buoy_ds
        self.name = f"{getattr(base_ds, 'name', 'buoy')}_on_{getattr(buoy_ds, 'name', 'buoy')}"
        self._model_var_names = self._channel_labels(var_names, "var_names")
        self._model_units = self._channel_labels(units, "units")
        lat, lon = lat_lon_from_grid(self.base_ds.grid)
        self.base_ds_coords = np.stack((lat.flatten(), lon.flatten()), axis=1)
        self.lat_first = lat_first
        if not lat_first:
            self.base_ds_coords = self.base_ds_coords[:, ::-1]
        self.interpolator = InvDistTree_np(self.base_ds_coords)
        self.base_time_axis = base_time_axis
        self.buoy_time_axis = buoy_time_axis
        self.static = static
        if self.static:
            self.interpolator.set_queries(self.buoy_ds.coords, n_near=1)

    @staticmethod
    def _channel_labels(labels, name):
        if labels is None:
            return None
        if isinstance(labels, (str, bytes)):
            raise ValueError(f"{name} must be a sequence with one entry per model channel.")
        labels = tuple(labels)
        if not all(isinstance(label, str) and label for label in labels):
            raise ValueError(f"{name} must contain nonempty strings.")
        if name == "var_names" and len(set(labels)) != len(labels):
            raise ValueError("var_names must contain unique model channel names.")
        return labels

    def __getitem__(self, idx, base_time_axis=None, buoy_time_axis=None):
        buoy_time_axis = buoy_time_axis if buoy_time_axis is not None else self.buoy_time_axis
        base_time_axis = base_time_axis if base_time_axis is not None else self.base_time_axis
        data = self.base_ds[idx]
        if data is None:
            return None
        batch = None
        if not self.static:
            batch = self.buoy_ds[idx]
            batch_fields = ("bids", "datetimes", "var_names", "units", "coord_valid", "coord_times")
            if all(hasattr(batch, field) for field in batch_fields):
                return self._sample_batch(data, batch, base_time_axis=base_time_axis)

        # Preserve the original behavior for static and legacy coords-only inputs.
        data_type = type(data)
        meta = getattr(data, 'meta', None)
        shape = data.shape
        data = data.reshape(shape[:-2] + (-1,))
        if self.static:
            data = self.interpolator.interpolate_static(data, space_axis=-1)
        else:
            self.interpolator.set_queries(batch.coords, q_time_axis=buoy_time_axis, n_near=1)
            data = self.interpolator.interpolate_timevarying(data, time_axis=base_time_axis, space_axis=-1)
        if meta is not None:
            data = data_type(data, **meta)
        return data

    def _sample_batch(self, data, batch, *, base_time_axis):
        from libs.validation.validator import MetricField

        values = np.asarray(data)
        if values.ndim != 4:
            raise ValueError("BuoyBatch sampling requires a grid shaped (T,V,H,W) or (V,T,H,W).")
        if (
            not isinstance(base_time_axis, (int, np.integer))
            or isinstance(base_time_axis, bool)
            or not -values.ndim <= base_time_axis < values.ndim
            or base_time_axis % values.ndim not in (0, 1)
        ):
            raise ValueError("base_time_axis must identify one of the first two grid axes.")
        time_axis = base_time_axis % values.ndim
        variable_axis = 1 - time_axis
        bids = np.asarray(batch.bids)
        datetimes = np.asarray(batch.datetimes)
        coords = np.asarray(batch.coords)
        coord_times = np.asarray(batch.coord_times)
        if bids.ndim != 1 or datetimes.ndim != 1:
            raise ValueError("BuoyBatch bids and datetimes must be one-dimensional.")
        point_shape = (len(bids), len(datetimes))
        if coords.shape != point_shape + (2,):
            raise ValueError("BuoyBatch coords must have shape (N,T,2).")
        coord_valid = np.asarray(batch.coord_valid, dtype=bool)
        if coord_valid.shape != point_shape or coord_times.shape != point_shape:
            raise ValueError("BuoyBatch coordinate masks and times must have shape (N,T).")
        coord_valid = coord_valid & np.isfinite(coords).all(axis=-1)
        if values.shape[time_axis] != len(datetimes):
            raise ValueError("Model and BuoyBatch must contain the same number of time steps.")
        if np.prod(values.shape[-2:]) != len(self.base_ds_coords):
            raise ValueError("Model spatial dimensions do not match its grid coordinates.")

        metadata = dict(getattr(data, "meta", {}) or {})
        if "datetimes" in metadata:
            try:
                model_times = np.asarray(metadata["datetimes"], dtype="datetime64[ns]")
                query_times = datetimes.astype("datetime64[ns]")
            except (TypeError, ValueError) as exc:
                raise ValueError("Model datetimes must describe each grid time step.") from exc
            if model_times.shape != query_times.shape or not np.array_equal(model_times, query_times):
                raise ValueError("Model datetimes must match the BuoyBatch query times before interpolation.")
        for key, override in (("var_names", self._model_var_names), ("units", self._model_units)):
            labels = self._channel_labels(override if override is not None else metadata.get(key), key)
            if labels is not None:
                if len(labels) != values.shape[variable_axis]:
                    raise ValueError(f"{key} must describe all {values.shape[variable_axis]} model channels.")
                metadata[key] = labels
            else:
                metadata.pop(key, None)
        if "valid" in metadata:
            try:
                grid_valid = np.broadcast_to(np.asarray(metadata["valid"], dtype=bool), values.shape)
            except ValueError as exc:
                raise ValueError("Model valid mask must broadcast to the original grid field.") from exc
            values = np.where(grid_valid, values, np.nan)

        output_shape = values.shape[:2] + (len(bids),)
        if not np.any(coord_valid):
            sampled = np.full(output_shape, np.nan, dtype=np.result_type(values.dtype, np.float32))
        else:
            # KDTree rejects NaN queries. Invalid positions borrow a finite query
            # temporarily and are masked again after interpolation.
            queries = np.array(coords, copy=True)
            queries[~coord_valid] = coords[coord_valid][0]
            if not self.lat_first:
                queries = queries[..., ::-1]
            self.interpolator.set_queries(queries, q_time_axis=1, n_near=1)
            sampled = self.interpolator.interpolate_timevarying(
                values.reshape(values.shape[:2] + (-1,)),
                time_axis=time_axis, space_axis=-1,
            )
            point_mask = coord_valid.T[:, None, :] if time_axis == 0 else coord_valid.T[None, :, :]
            sampled = np.where(point_mask, sampled, np.nan)

        dims = ("time", "variable", "buoy") if time_axis == 0 else ("variable", "time", "buoy")
        metadata.update(
            dims=dims,
            bids=bids.copy(),
            datetimes=datetimes.copy(),
            coords=coords.copy(),
            coord_valid=coord_valid.copy(),
            coord_times=coord_times.copy(),
            valid=np.isfinite(sampled),
            buoy_sampling={"var_names": tuple(batch.var_names), "units": tuple(batch.units)},
        )
        return MetricField(sampled, **metadata)

    def __getattr__(self, name):
        # Delegate metadata like src_grid, grid, seq_len, etc.
        return getattr(self.base_ds, name)


def lat_lon_from_grid(grid):
    lat_names = ['lat', 'latitude', 'XLAT', 'Latitude']
    lon_names = ['lon', 'long', 'longitude', 'XLON', 'XLONG', 'Longitude']

    def _looks_like_array(x):
        # Reject addict auto-created empties / dicts
        if isinstance(x, Mapping):
            return False
        # Accept numpy arrays and array-like objects
        return isinstance(x, np.ndarray) or hasattr(x, "shape")

    def _get(obj, names):
        # 1) Mapping path (safe for addict.Dict)
        if isinstance(obj, Mapping):
            for name in names:
                if name in obj:              # does NOT create in addict
                    val = obj[name]
                    if _looks_like_array(val):
                        return val

        # 2) Attribute path (for non-mapping grid objects)
        for name in names:
            try:
                val = getattr(obj, name)
            except AttributeError:
                continue
            if _looks_like_array(val):
                return val

        return None

    lat = _get(grid, lat_names)
    lon = _get(grid, lon_names)
    return lat, lon


class ConstantDataset(torch.utils.data.Dataset):
    """
    Dataset-wrapper over a constant array.

    Useful when the mask is static (for example, a fixed regional mask).
    Ignores index/date and always returns the same value.
    """
    def __init__(self, value, length=None):
        self.value = np.asarray(value)
        self._length = 0 if length is None else int(length)

    def __getitem__(self, index, *args, **kwargs):
        return self.value

    def __len__(self):
        return self._length


class MaskedDataset(torch.utils.data.Dataset):
    """
    Wraps another dataset and multiplies its output by a mask.

    Parameters
    ----------
    dataset : Dataset
        Main dataset returning arrays, typically of shape (T, C, H, W).
    mask : Dataset | np.ndarray
        Either:
          - a dataset indexed by the same date/index as `dataset`, or
          - a constant array mask.
        If a constant array is passed, it is wrapped into ConstantDataset.
    dtype : np.dtype | None
        Optional dtype for the result.
    copy : bool
        If True, explicitly copies the source array before masking.

    Notes
    -----
    Common supported broadcasting cases:
      data (T, C, H, W), mask (H, W)      -> expanded to (1, 1, H, W)
      data (T, C, H, W), mask (T, H, W)   -> expanded to (T, 1, H, W)
      data (T, C, H, W), mask (C, H, W)   -> expanded to (1, C, H, W)
      data (T, C, H, W), mask (T, C, H, W)-> used as is

    If you need a more exotic layout, just return a mask already broadcastable
    to the data shape.
    """
    def __init__(self, dataset, mask, dtype=None, copy=False):
        self.dataset = dataset
        self.mask_dataset = (
            mask if isinstance(mask, Dataset)
            else ConstantDataset(mask, length=len(dataset))
        )
        self._length = len(dataset)
        self.dtype = dtype
        self.copy = copy

    @staticmethod
    def _prepare_mask(mask, data):
        mask = np.asarray(mask)
        data = np.asarray(data)

        # Most common geospatial case: data (T, C, H, W), mask (H, W)
        if mask.ndim == 2 and data.ndim >= 4 and mask.shape == data.shape[-2:]:
            mask = mask[None, None, ...]

        # Time-dependent spatial mask: data (T, C, H, W), mask (T, H, W)
        elif mask.ndim == 3 and data.ndim >= 4 \
                and mask.shape[0] == data.shape[0] \
                and mask.shape[-2:] == data.shape[-2:]:
            mask = mask[:, None, ...]

        # Channel-dependent mask: data (T, C, H, W), mask (C, H, W)
        elif mask.ndim == 3 and data.ndim >= 4 and mask.shape == data.shape[-3:]:
            mask = mask[None, ...]

        # Station-like case: data (T, N, V), mask (N,)
        elif mask.ndim == 1 and data.ndim == 3 and mask.shape[0] == data.shape[1]:
            mask = mask[None, :, None]

        try:
            np.broadcast_shapes(mask.shape, data.shape)
        except ValueError as e:
            raise ValueError(
                f"Mask with shape {mask.shape} cannot be broadcast to data "
                f"with shape {data.shape}"
            ) from e

        return mask

    def _get_mask(self, index, *args, **kwargs):
        try:
            return self.mask_dataset.__getitem__(index, *args, **kwargs)
        except TypeError:
            # fallback for datasets that only accept index/date
            return self.mask_dataset[index]

    def __getitem__(self, index, *args, **kwargs):
        data = self.dataset.__getitem__(index, *args, **kwargs)
        if data is None:
            return None

        mask = self._get_mask(index, *args, **kwargs)
        if mask is None:
            return data

        # data = np.array(data, copy=self.copy)
        mask = self._prepare_mask(mask, data)

        out = data * mask
        if self.dtype is not None:
            out = out.astype(self.dtype, copy=False)
        return out

    def __len__(self):
        return self._length

    def __getattr__(self, name):
        # Delegate metadata like src_grid, grid, seq_len, etc.
        return getattr(self.dataset, name)
