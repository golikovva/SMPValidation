from abc import abstractmethod  # For defining abstract base classes and methods
from datetime import datetime  # For handling date and time operations

import numpy as np  # For numerical operations
import pyproj  # For cartographic projections and transformations
import xarray as xr  # For handling multi-dimensional datasets

from libs.validation import Grid  # Custom Grid class from boreylib library
from libs.validation.datasets.base import Dataset  # Base Dataset class from boreylib


class AmsrJaxaHsiDataset(Dataset):
    """
    Base class for AMSR JAXA HSI datasets, which defines common grid creation,
    data processing, and extraction methods for handling HSI data files.
    """

    def _create_grid(self):
        """
        Creates a stereographic grid with specified latitude and longitude dimensions.

        Returns:
            Grid: A Grid object containing lat/lon data.
        """
        # Define x and y coordinates in a stereographic projection
        x = np.linspace(-3_850_000, 3_750_000, self._grid_shape[0])
        y = np.linspace(5_850_000, -5_350_000, self._grid_shape[1])
        xx, yy = np.meshgrid(x, y)

        # Convert stereographic coordinates to lat/lon using pyproj
        stereo_crs = pyproj.CRS.from_epsg(3413)
        plate_caree_crs = pyproj.CRS.from_epsg(4326)
        transformer = pyproj.Transformer.from_crs(stereo_crs, plate_caree_crs)
        lat, lon = transformer.transform(xx, yy)

        # Create and return a Grid object with the calculated lat/lon values
        grid = Grid(lat, lon)
        return grid

    def _process_field(self, field):
        """
        Processes the raw field data by applying transformations and masking invalid values.

        Args:
            field (np.array): Raw data field to process.

        Returns:
            np.array: Processed data field with time, latitude, and longitude dimensions.
        """
        field = field.astype(np.float32)  # Convert to float32 for consistency
        field = field.transpose(2, 0, 1)  # Reorder dimensions to (time, lat, lon)
        field[field <= -32_767.0] = np.nan  # Mask invalid values with NaN
        return field

    def _extract_data(self, file, load_fn=xr.open_dataset):
        """
        Extracts geophysical data from an HDF5 file and applies scaling.

        Args:
            file (str): Path to the data file.
            load_fn (callable): Function to load the dataset (default is xarray's open_dataset).

        Returns:
            np.array: Extracted and processed data array.
        """
        ds = load_fn(file)  # Load the dataset
        entry = ds.variables['Geophysical Data']  # Extract geophysical data
        data = self._process_field(entry.values)  # Process the raw data
        data = data[:, None, :, :]  # (time, channel, lat, lon)
        data = data * float(entry.attrs['SCALE FACTOR'])  # Apply scaling factor
        return data

    @staticmethod
    def _parse_date(file):
        """
        Parses the date from the filename of the dataset.

        Args:
            file (str): Path to the data file.

        Returns:
            datetime.date: Extracted date from the filename.
        """
        date_part = file.name.split('_')[1]  # Extract date part from filename
        date = datetime.strptime(date_part, '%Y%m%d').date()  # Parse the date
        return date

    @property
    @abstractmethod
    def _grid_shape(self):
        """
        Abstract property to define the shape of the grid.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError


class AmsrJaxaHsi25kmDataset(AmsrJaxaHsiDataset):
    """
    Dataset class for AMSR JAXA HSI 25km resolution data.
    """

    @property
    def _files_template(self):
        """
        Defines the file naming template for 25km HSI data.

        Returns:
            str: File path template for 25km data.
        """
        return 'HSI/v100/L3/**/GW1AM2_*_01D_PNM?_L3RGHSILW1100100.h5'

    @property
    def _grid_shape(self):
        """
        Defines the grid shape for 25km resolution data.

        Returns:
            tuple: Shape of the grid (lat, lon).
        """
        return 304, 448  # 25km resolution grid size


class AmsrJaxaHsi10kmDataset(AmsrJaxaHsiDataset):
    """
    Dataset class for AMSR JAXA HSI 10km resolution data.
    """

    @property
    def _files_template(self):
        """
        Defines the file naming template for 10km HSI data.

        Returns:
            str: File path template for 10km data.
        """
        return 'HSI/v100/L3/**/GW1AM2_*_01D_PNM?_L3RGHSIHW1100100.h5'

    @property
    def _grid_shape(self):
        """
        Defines the grid shape for 10km resolution data.

        Returns:
            tuple: Shape of the grid (lat, lon).
        """
        return 760, 1120  # 10km resolution grid size


class AmsrJaxaSimRDataset(Dataset):
    """
    Dataset class for AMSR JAXA SIM-R (Radiometer simulation) data.
    """

    def _create_grid(self):
        """
        Creates a grid from a pre-existing SIM-R lat/lon file.

        Returns:
            Grid: A Grid object containing lat/lon data.
        """
        grid_path = self.path / 'SIM_R/SIM_R_latlon.dat'  # Path to lat/lon data file
        raw_grid = np.fromfile(grid_path, dtype=np.float32).reshape(2, 448, 304)  # Load and reshape data
        raw_grid[raw_grid == -32768.0] = -32767.0  # Fix invalid grid values
        lat, lon = raw_grid
        lat = self._process_field(lat)[0]  # Process latitude data
        lon = self._process_field(lon)[0]  # Process longitude data
        grid = Grid(lat, lon)
        return grid

    @property
    def _files_template(self):
        """
        Defines the file naming template for SIM-R data.

        Returns:
            str: File path template for SIM-R data.
        """
        return 'SIM_R/v101/L3/**/GW1AM2_*_01D_PNMB_L3RGSIMLR1101101.h5'

    def _process_field(self, field):
        """
        Processes the raw SIM-R field by masking invalid values and subsetting the region.

        Args:
            field (np.array): Raw data field to process.

        Returns:
            np.array: Processed field with time, latitude, and longitude dimensions.
        """
        region_mask = (field != -32_767)  # Create a mask for valid data
        i_range = np.arange(region_mask.shape[0])[region_mask.any(axis=1)]  # Find the latitude bounds
        i_slice = slice(i_range.min(), i_range.max() + 1)
        j_range = np.arange(region_mask.shape[1])[region_mask.any(axis=0)]  # Find the longitude bounds
        j_slice = slice(j_range.min(), j_range.max() + 1)
        region = (i_slice, j_slice)
        assert region_mask[region[0], region[1]].all(), 'masked region is not rectangular'  # Ensure the mask is rectangular

        field = field[region[0], region[1]]  # Subset the field
        field = field[::2, ::2]  # Downsample the field
        field = field.astype(np.float32)
        field[field == -32_768.0] = np.nan  # Mask invalid values
        field = field[None, :, :]  # (time, lat, lon)
        return field

    def _extract_data(self, file, load_fn=xr.open_dataset):
        """
        Extracts geophysical data from SIM-R files, processes the u and v components.

        Args:
            file (str): Path to the data file.
            load_fn (callable): Function to load the dataset (default is xarray's open_dataset).

        Returns:
            np.array: Extracted and processed data array.
        """
        ds = load_fn(file)  # Load the dataset
        entry = ds.variables['Geophysical Data EN']
        ufield, vfield = entry.values.transpose(2, 0, 1)  # Extract and transpose u and v components
        data = np.stack([
            self._process_field(ufield),  # Process u-component
            self._process_field(vfield),  # Process v-component
        ], axis=1)  # (time, channel, lat, lon)
        data = data * float(entry.attrs['SCALE FACTOR'])  # Apply scaling factor
        return data

    @staticmethod
    def _parse_date(file):
        """
        Parses the date from the filename of the SIM-R dataset.

        Args:
            file (str): Path to the data file.

        Returns:
            datetime.date: Extracted date from the filename.
        """
        date_part = file.name.split('_')[1]  # Extract date part from filename
        date = datetime.strptime(date_part, '%Y%m%d').date()  # Parse the date
        return date


class AmsrJaxaSimYDataset(Dataset):
    """
    Dataset class for AMSR JAXA SIM-Y (Yearly simulation) data.
    """

    def _create_grid(self):
        """
        Loads the grid from an existing SIM-Y data file.

        Returns:
            Grid: A Grid object containing lat/lon data.
        """
        grid_path = sorted(self.path.glob(self._files_template))[0]  # Find the first matching file
        print(f'loading grid from {grid_path}')
        ds = xr.open_dataset(grid_path)
        lat = ds.variables['lat'].values  # Extract latitude data
        lon = ds.variables['lon'].values  # Extract longitude data
        grid = Grid(lat, lon)
        return grid

    @property
    def _files_template(self):
        """
        Defines the file naming template for SIM-Y data.

        Returns:
            str: File path template for SIM-Y data.
        """
        return 'SIM_Y/v100/L3/**/GW1AM2_*_01D_PNMB_L3RGSIMPY1100100.h5'

    def _process_field(self, field):
        """
        Processes the raw SIM-Y field by masking invalid values.

        Args:
            field (np.array): Raw data field to process.

        Returns:
            np.array: Processed field with time, latitude, and longitude dimensions.
        """
        field = field.astype(np.float32)
        field[field == -32_768.0] = np.nan  # Mask invalid values
        field = field[None, :, :]  # (time, lat, lon)
        return field

    def _extract_data(self, file, load_fn=xr.open_dataset):
        """
        Extracts geophysical data from SIM-Y files, processes the u and v components.

        Args:
            file (str): Path to the data file.
            load_fn (callable): Function to load the dataset (default is xarray's open_dataset).

        Returns:
            np.array: Extracted and processed data array.
        """
        ds = load_fn(file)  # Load the dataset
        ufield = ds.variables['ve'].values  # Extract the u-component
        vfield = ds.variables['vn'].values  # Extract the v-component
        data = np.stack([
            self._process_field(ufield),  # Process u-component
            self._process_field(vfield),  # Process v-component
        ], axis=1)  # (time, channel, lat, lon)
        return data

    @staticmethod
    def _parse_date(file):
        """
        Parses the date from the filename of the SIM-Y dataset.

        Args:
            file (str): Path to the data file.

        Returns:
            datetime.date: Extracted date from the filename.
        """
        date_part = file.name.split('_')[1]  # Extract date part from filename
        date = datetime.strptime(date_part, '%Y%m%d').date()  # Parse the date
        return date
