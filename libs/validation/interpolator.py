from contextlib import ExitStack

import esmpy  # ESMF Python bindings for regridding and interpolation
import numpy as np  # For numerical operations with arrays

from .interpolation_cache import _create_regrid


class Interpolator:
    """
    This class performs interpolation from one grid to another using the Earth System Modeling Framework (ESMF).

    Attributes:
        src_grid: Source grid object containing the grid to interpolate from.
        dst_grid: Destination grid object containing the grid to interpolate to.
        regrid: ESMF regrid object to handle the interpolation.
        dst_region: Boolean mask array for the destination grid, indicating valid interpolated values.
    """

    def __init__(self, src_grid, dst_grid):
        """
        Initializes the Interpolator with source and destination grids.

        Args:
            src_grid: Source grid object containing lat/lon information.
            dst_grid: Destination grid object containing lat/lon information.
        """
        self.src_grid = src_grid  # Source grid for interpolation
        self.dst_grid = dst_grid  # Destination grid for interpolation

        self.regrid = None  # ESMF Regrid object (initialized in `initialize` method)
        self.dst_region = None  # Boolean mask for valid interpolation regions

    def initialize(self, *, cache_dir=None):
        """Build an operator, or reuse disk weights for the same pair of grids.

        ``cache_dir`` accepts a path to a shared weights directory. ``None``
        disables disk caching. Cache failures fall back to ordinary regridding.
        Reinitialization releases the previous operator; the grids stay owned
        by the caller.
        """
        self.destroy()
        regrid = None
        try:
            with ExitStack() as resources:
                src_field = esmpy.Field(grid=self.src_grid.grid)
                resources.callback(src_field.destroy)
                dst_field = esmpy.Field(grid=self.dst_grid.grid)
                resources.callback(dst_field.destroy)
                src_field.data[:] = 0.0
                dst_field.data[:] = np.nan

                src_scale = self.src_grid.cell_areas().mean()
                dst_scale = self.dst_grid.cell_areas().mean()
                if src_scale < dst_scale:
                    method = esmpy.api.constants.RegridMethod.CONSERVE
                    print('Using conservative regrid method')
                else:
                    method = esmpy.api.constants.RegridMethod.BILINEAR
                    print('Using bilinear regrid method')

                options = {
                    'regrid_method': method,
                    'unmapped_action': esmpy.api.constants.UnmappedAction.IGNORE,
                }
                regrid = _create_regrid(
                    self.src_grid, self.dst_grid, src_field, dst_field,
                    options, cache_dir,
                )

                # Constructors may change field data. Probe coverage explicitly
                # after both loading and building weights, leaving unmapped NaNs.
                src_field.data[:] = 1.0
                dst_field.data[:] = np.nan
                regrid(src_field, dst_field, zero_region=esmpy.Region.SELECT)
                dst_region = ~np.isnan(dst_field.data)
        except BaseException:
            if regrid is not None:
                regrid.destroy()
            raise

        self.regrid = regrid
        self.dst_region = dst_region

    def destroy(self):
        """Release the operator without destroying the caller's grids."""
        regrid = self.regrid
        self.regrid = None
        self.dst_region = None
        if regrid is not None:
            regrid.destroy()

    def __call__(self, field):
        """
        Interpolates a given field from the source grid to the destination grid.

        Args:
            field: A 2D numpy array containing data on the source grid to be interpolated.

        Returns:
            A 2D numpy array containing the interpolated data on the destination grid, with NaN values outside valid regions.
        """
        assert self.regrid is not None, 'Interpolator must be initialized before use'  # Ensure regrid object is initialized

        with ExitStack() as resources:
            src_field = esmpy.Field(grid=self.src_grid.grid)
            resources.callback(src_field.destroy)
            src_field.data[:] = field
            dst_field = esmpy.Field(grid=self.dst_grid.grid)
            resources.callback(dst_field.destroy)
            dst_field.data[:] = np.nan

            interpolated_field = self.regrid(src_field, dst_field).data.copy()
            interpolated_field[~self.dst_region] = np.nan
            return interpolated_field
