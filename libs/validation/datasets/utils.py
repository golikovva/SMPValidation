import numpy as np
import geopandas as gpd
from shapely import contains_xy
from datetime import timedelta
from typing import Union
from libs.validation.datasets.base import Dataset




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
    
def sic_to_sie(sic_dataset_class):
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
            
            # Apply 15% threshold to convert to binary ice extent
            sie = (field >= 15).astype(np.float32)
            
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