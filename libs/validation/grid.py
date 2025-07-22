import esmpy  # Earth System Modeling Framework (ESMF) Python bindings
import global_land_mask as glm  # Library to check if a point is over land or ocean
import numpy as np  # Library for numerical computations
import pyproj  # Library for cartographic projections and coordinate transformations
import scipy  # Scientific computing library
import xarray as xr  # Library for working with multi-dimensional arrays and netCDF files


class Grid:
    """
    Class representing a 2D grid of latitude and longitude coordinates.

    Attributes:
        earth_radius (float): The radius of the Earth in kilometers (used for calculating cell areas).
        shape (tuple): Shape of the latitude and longitude arrays.
        grid (esmpy.Grid): ESMF grid object used for defining grid structure and properties.
    """
    earth_radius = 6371.0  # Earth's radius in kilometers for area calculations

    def __init__(self, lat, lon):
        """
        Initializes the Grid with latitude and longitude arrays.

        Args:
            lat (np.array): 2D array of latitude values.
            lon (np.array): 2D array of longitude values.
        """
        assert lat.shape == lon.shape, 'lat and lon must have the same shape'  # Ensure lat/lon arrays match in shape
        self.shape = lat.shape  # Store the shape of lat/lon arrays

        # Initialize ESMF Grid with center and corner stagger locations using spherical degrees coordinate system
        self.grid = esmpy.Grid(
            np.array(self.shape),
            staggerloc=[esmpy.StaggerLoc.CENTER, esmpy.StaggerLoc.CORNER],
            coord_sys=esmpy.CoordSys.SPH_DEG
        )

        # Assign center coordinates (longitude and latitude) to the grid
        self.grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CENTER)[:] = lon
        self.grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CENTER)[:] = lat

        # Estimate and assign corner coordinates for the grid cells
        lat_corners, lon_corners = self._estimate_cell_corners(lat, lon)
        self.grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CORNER)[:] = lon_corners
        self.grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CORNER)[:] = lat_corners

    @property
    def lat(self):
        """
        Returns the latitude values at the center of the grid cells.
        """
        return self.grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CENTER)[:]

    @property
    def lon(self):
        """
        Returns the longitude values at the center of the grid cells.
        """
        return self.grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CENTER)[:]

    @property
    def lat_corners(self):
        """
        Returns the latitude values at the corners of the grid cells.
        """
        return self.grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CORNER)[:]

    @property
    def lon_corners(self):
        """
        Returns the longitude values at the corners of the grid cells.
        """
        return self.grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CORNER)[:]

    def cell_areas(self):
        """
        Calculates the areas of each cell in the grid, using the Earth's radius.

        Returns:
            np.array: 2D array of cell areas in square kilometers.
        """
        field = esmpy.Field(self.grid)  # Create an ESMF Field associated with the grid
        field.get_area()  # Get the areas of the cells in radians^2
        areas = field.data * self.earth_radius ** 2  # Convert to km^2 using Earth's radius
        return areas

    def land_mask(self):
        """
        Creates a land mask for the grid, where land is True and ocean is False.

        Returns:
            np.array: 2D boolean array, True if the cell is over land, False if over ocean.
        """
        mask = glm.is_land(self.lat, self.lon)  # Use global land mask to determine land or ocean
        return mask

    def as_netcdf(self, title):
        """
        Saves the grid data (latitude and longitude) as a NetCDF file using xarray.

        Args:
            title (str): Title to be added as global metadata in the NetCDF file.

        Returns:
            xarray.Dataset: A dataset containing the latitude and longitude information.
        """
        # Create xarray DataArrays for latitude and longitude with dimensions y (rows) and x (columns)
        lat = xr.DataArray(self.lat, dims=['y', 'x'], name='lat')
        lon = xr.DataArray(self.lon, dims=['y', 'x'], name='lon')

        # Combine latitude and longitude into an xarray Dataset
        ds = xr.Dataset({'lat': lat, 'lon': lon})

        # Add attributes to describe the latitude and longitude variables
        ds['lat'].attrs['units'] = 'degrees'
        ds['lat'].attrs['description'] = 'Latitude'
        ds['lon'].attrs['units'] = 'degrees'
        ds['lon'].attrs['description'] = 'Longitude'

        # Add global attribute to describe the dataset
        ds.attrs['title'] = title
        return ds

    @staticmethod
    def _estimate_cell_corners(lat, lon):
        """
        Estimates the latitude and longitude values at the corners of each grid cell.

        Args:
            lat (np.array): 2D array of latitude values.
            lon (np.array): 2D array of longitude values.

        Returns:
            tuple: Two 2D arrays of latitudes and longitudes at the corners of the grid cells.
        """
        # Define projections: stereographic for Arctic region and Plate Carrée for global lat/lon
        stereo_crs = pyproj.CRS.from_epsg(3413)  # Polar stereographic projection (EPSG 3413)
        plate_caree_crs = pyproj.CRS.from_epsg(4326)  # Plate Carrée projection (EPSG 4326)

        # Transform latitude and longitude into stereographic coordinates
        transformer = pyproj.Transformer.from_crs(plate_caree_crs, stereo_crs)
        xx, yy = transformer.transform(lat, lon)

        # Estimate the corners of the transformed grid
        xx_corners = Grid._estimate_array_corners(xx)
        yy_corners = Grid._estimate_array_corners(yy)

        # Convert the corners back to lat/lon
        transformer = pyproj.Transformer.from_crs(stereo_crs, plate_caree_crs)
        lat_corners, lon_corners = transformer.transform(xx_corners, yy_corners)
        return lat_corners, lon_corners

    @staticmethod
    def _estimate_array_corners(arr):
        """
        Estimates the corner values of a 2D array by extrapolating its boundary values.

        Args:
            arr (np.array): 2D array representing values (e.g., lat or lon).

        Returns:
            np.array: 2D array representing the corners of the input array.
        """
        assert len(arr.shape) == 2, 'array to extrapolate must be 2D'
        assert arr.shape[0] > 1 and arr.shape[1] > 1, 'array must have at least 2 elements per dimension'

        # Pad the array with extra elements on all sides
        extr_arr = np.pad(arr, 1)

        # Perform linear extrapolation on the array's boundaries
        extr_arr[0, :] = 2 * extr_arr[1, :] - extr_arr[2, :]
        extr_arr[:, -1] = 2 * extr_arr[:, -2] - extr_arr[:, -3]
        extr_arr[-1, :] = 2 * extr_arr[-2, :] - extr_arr[-3, :]
        extr_arr[:, 0] = 2 * extr_arr[:, 1] - extr_arr[:, 2]

        # Use a simple averaging kernel to estimate corner values using convolution
        corners = scipy.signal.convolve2d(extr_arr, 0.25 * np.ones((2, 2)), mode='valid')
        return corners


class BarentsKaraGrid(Grid):
    """
    A specialized grid for the Barents and Kara Seas region with a default resolution.
    Inherits from the Grid class.
    """

    def __init__(self, resolution=1.0):
        """
        Initializes a grid for the Barents and Kara Seas using the provided resolution.

        Args:
            resolution (float): The resolution of the grid (default is 1.0 km).
        """
        # Create grid coordinates (x, y) in stereographic projection with the specified resolution
        x = np.linspace(1_200_000, 2_700_000, round(1500 / resolution))
        y = np.linspace(1_000_000, -800_000, round(1800 / resolution))
        xx, yy = np.meshgrid(x, y)  # Create 2D meshgrid from x and y coordinates

        # Transform stereographic coordinates back to latitude and longitude
        stereo_crs = pyproj.CRS.from_epsg(3413)
        plate_caree_crs = pyproj.CRS.from_epsg(4326)
        transformer = pyproj.Transformer.from_crs(stereo_crs, plate_caree_crs)
        lat, lon = transformer.transform(xx, yy)

        # Call the parent Grid class constructor
        super().__init__(lat, lon)


class PseudoNemoGrid(Grid):
    """
    A specialized grid with a rotated coordinate system for a specific region (Pseudo NEMO grid).
    Inherits from the Grid class.
    """

    def __init__(self, resolution=1.0):
        """
        Initializes a Pseudo NEMO grid with a rotated coordinate system.

        Args:
            resolution (float): The resolution of the grid (default is 1.0 km).
        """
        # Define grid coordinates in stereographic projection
        x = np.linspace(1_385_000, 2_335_000, round(950 / resolution))
        y = np.linspace(-1_565_000, -275_000, round(1290 / resolution))

        # Define rotation matrix to rotate grid by 32 degrees
        theta = np.radians(32.0)
        pregrid = np.meshgrid(x, y)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        xx, yy = np.einsum('ij, jmn -> imn', rot, pregrid)  # Apply rotation

        # Transform rotated coordinates back to latitude and longitude
        stereo_crs = pyproj.CRS.from_epsg(3413)
        plate_caree_crs = pyproj.CRS.from_epsg(4326)
        transformer = pyproj.Transformer.from_crs(stereo_crs, plate_caree_crs)
        lat, lon = transformer.transform(xx, yy)

        # Call the parent Grid class constructor
        super().__init__(lat, lon)
