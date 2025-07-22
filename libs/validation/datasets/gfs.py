from datetime import datetime
import pygrib
import numpy as np
from torch.utils.data import Dataset

from libs.validation import Grid
from libs.validation.datasets.base import (
    Dataset,
)


class GFSDataset(Dataset):
    def __init__(self, path, variables, dst_grid=None, average_times=None, name=None):
        self.variables = variables
        super().__init__(path, dst_grid, average_times, name)

    def _create_grid(self):
        """
        Creates the grid by loading latitude and longitude from the dataset files.

        Returns:
            Grid: A Grid object containing latitude and longitude arrays.
        """
        # Find the first matching file to load the grid information
        grid_path = sorted(self.path.glob(self._files_template))[0]
        print(f'loading grid from {grid_path}')
        with pygrib.open(grid_path) as ds:
            # Read the first message (assuming grid is consistent across messages)
            grb = ds.read(1)[0]
            
            # Extract latitude/longitude arrays (2D grids)
            lat, lon = grb.latlons()

        # Create and return a Grid object
        grid = Grid(lat, lon)
        return grid

    def _extract_data(self, filename):
        """
        Extracts wind data from the given file.

        Args:
            file (str): Path to the data file.
            load_fn (callable, optional): Function to load the file.

        Returns:
            np.array: Extracted data.
        """
        out = []
        with pygrib.open(filename) as ds:
            for i in self.variables:
                try:
                    layer = [k for k in ds.select(name=i)]
                except ValueError:
                    print('Value Error')
                # check if param unique in grib
                if len(layer) != 1:
                    print(f'file has {len(layer)} params, must be 1')
                layer = layer[0]
                geophysical_data, lats, lons = layer.data() # lat1=62.22, lat2=78.75, lon1=22.24, lon2=75.81)
                out.append(geophysical_data)
        return np.stack(out)[None]
    
    def _process_field(self, field):
        """
        Processes the wind field without additional transformations.

        Args:
            field (np.array): Raw wind data.

        Returns:
            np.array: Processed wind data.
        """
    
        return field  # No processing required for wind
    
    @staticmethod
    def _parse_date(file):
        """
        Parses the date from the filename of the GFS dataset files.

        Args:
            file (Path): The file object containing the filename.

        Returns:
            datetime.date: The parsed date from the filename.
        """
        # Extract the date part from the filename and parse it
        date_part = file.stem.split('.')[-2]
        date = datetime.strptime(date_part, '%Y%m%d%H').date()  # Parse as 'YYYYMMDDHH'
        return date
    
    @property
    def _files_template(self):
        """
        Returns the file template pattern for locating NEMO model dataset files.

        Returns:
            str: The file path template.
        """
        return 'gfs.*.f*.grib2'
    