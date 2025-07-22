import esmpy  # ESMF Python bindings for regridding and interpolation
import numpy as np  # For numerical operations with arrays


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

    def initialize(self):
        """
        Initializes the ESMF regridding method based on the size of the source and destination grids.
        Determines whether to use conservative or bilinear interpolation based on cell area comparison.
        """

        # Create empty source and destination fields using the grids
        src_field = esmpy.Field(grid=self.src_grid.grid)
        src_field.data[:] = 0.0  # Initialize source field with zeros

        dst_field = esmpy.Field(grid=self.dst_grid.grid)
        dst_field.data[:] = np.nan  # Initialize destination field with NaNs

        # Calculate the average cell area for source and destination grids
        src_scale = self.src_grid.cell_areas().mean()
        dst_scale = self.dst_grid.cell_areas().mean()

        # Choose regridding method based on relative grid cell sizes
        if src_scale < dst_scale:
            regrid_method = esmpy.api.constants.RegridMethod.CONSERVE  # Use conservative interpolation for smaller source grid cells
            print('Using conservative regrid method')
        else:
            regrid_method = esmpy.api.constants.RegridMethod.BILINEAR  # Use bilinear interpolation for larger source grid cells
            print('Using bilinear regrid method')

        # Initialize ESMF regrid object with chosen method and ignoring unmapped regions
        self.regrid = esmpy.Regrid(
            src_field,
            dst_field,
            regrid_method=regrid_method,
            unmapped_action=esmpy.api.constants.UnmappedAction.IGNORE  # Ignore areas with no overlap
        )

        # Create a boolean mask to indicate valid regions in the destination grid
        self.dst_region = ~np.isnan(dst_field.data)

    def __call__(self, field):
        """
        Interpolates a given field from the source grid to the destination grid.

        Args:
            field: A 2D numpy array containing data on the source grid to be interpolated.

        Returns:
            A 2D numpy array containing the interpolated data on the destination grid, with NaN values outside valid regions.
        """
        assert self.regrid is not None, 'Interpolator must be initialized before use'  # Ensure regrid object is initialized

        # Create a new source field and assign input data to it
        src_field = esmpy.Field(grid=self.src_grid.grid)
        src_field.data[:] = field

        # Create a destination field and initialize it with NaN values
        dst_field = esmpy.Field(grid=self.dst_grid.grid)
        dst_field.data[:] = np.nan

        # Perform the regridding (interpolation) and copy the result
        interpolated_field = self.regrid(src_field, dst_field).data.copy()

        # Mask out regions in the destination field that are invalid (i.e., no corresponding source data)
        interpolated_field[~self.dst_region] = np.nan

        # Clean up the ESMF fields
        src_field.destroy()
        dst_field.destroy()

        return interpolated_field  # Return the interpolated field
