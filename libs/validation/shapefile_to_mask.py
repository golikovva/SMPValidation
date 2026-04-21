from __future__ import annotations

from pathlib import Path
from typing import Mapping, Union, Literal
import tempfile
import zipfile
import urllib.request

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from matplotlib.path import Path as MplPath
from pyproj import CRS, Transformer
from libs.validation.visualization import lat_lon_from_grid


# Direct Natural Earth download URLs (10m physical vectors)
NE_URLS = {
    "land": "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_land.zip",
    "ocean": "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_ocean.zip",
}

PathLike = Union[str, Path]


def get_default_arctic_crs(lon_0: float = 0.0) -> CRS:
    """
    A robust planar CRS for Arctic domains that include the pole / antimeridian.

    LAEA centered on the North Pole is a good default for topology operations.
    """
    return CRS.from_proj4(
        f"+proj=laea +lat_0=90 +lon_0={lon_0} +datum=WGS84 +units=m +no_defs"
    )


def _grid_lat_lon_2d(grid):
    """
    Uses the user's lat_lon_from_grid(grid) helper and returns 2D lat/lon arrays.
    """
    lat, lon = lat_lon_from_grid(grid)

    if lat is None or lon is None:
        raise ValueError("Could not extract latitude/longitude from grid.")

    lat = np.asarray(lat).squeeze()
    lon = np.asarray(lon).squeeze()

    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    elif lat.ndim == 2 and lon.ndim == 2:
        if lat.shape != lon.shape:
            raise ValueError(
                f"2D lat/lon shapes do not match: lat={lat.shape}, lon={lon.shape}"
            )
        lat2d, lon2d = lat, lon
    else:
        raise ValueError(
            "lat/lon must be either both 1D or both 2D after squeeze(). "
            f"Got lat.ndim={lat.ndim}, lon.ndim={lon.ndim}"
        )

    return lat2d, lon2d


def _outer_boundary_ring_from_grid(lat2d: np.ndarray, lon2d: np.ndarray) -> np.ndarray:
    """
    Build an ordered outer boundary ring from the perimeter of a 2D grid.

    Returns
    -------
    ring_lonlat : np.ndarray of shape (N, 2)
        Boundary ring in (lon, lat) order, closed.
    """
    if lat2d.ndim != 2 or lon2d.ndim != 2:
        raise ValueError("lat2d and lon2d must both be 2D arrays.")
    if lat2d.shape != lon2d.shape:
        raise ValueError("lat2d and lon2d must have the same shape.")

    nrows, ncols = lat2d.shape
    if nrows < 2 or ncols < 2:
        raise ValueError("Grid must be at least 2x2 to build a boundary polygon.")

    top = np.column_stack([lon2d[0, :], lat2d[0, :]])
    right = np.column_stack([lon2d[1:, -1], lat2d[1:, -1]])
    bottom = np.column_stack([lon2d[-1, -2::-1], lat2d[-1, -2::-1]])
    left = np.column_stack([lon2d[-2:0:-1, 0], lat2d[-2:0:-1, 0]])

    ring = np.vstack([top, right, bottom, left])

    if np.isnan(ring).any():
        raise ValueError(
            "NaNs found on the grid boundary. "
            "Trim or fill the grid perimeter before creating the polygon."
        )

    # remove consecutive duplicates
    if len(ring) > 1:
        keep = np.ones(len(ring), dtype=bool)
        keep[1:] = np.any(np.diff(ring, axis=0) != 0, axis=1)
        ring = ring[keep]

    # close ring
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])

    if len(ring) < 4:
        raise ValueError("Boundary ring is too short to form a polygon.")

    return ring


def _project_lonlat_to_xy(
    lon: np.ndarray,
    lat: np.ndarray,
    dst_crs: CRS,
    src_crs: CRS = CRS.from_epsg(4326),
):
    """
    Project lon/lat arrays to x/y.
    """
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def make_grid_boundary_shapefile(
    grid,
    out_shapefile: PathLike,
    *,
    out_crs: CRS | str | None = None,
    fix_geometry: bool = True,
    return_gdf: bool = False,
):
    """
    Create a boundary shapefile from a grid, robust for Arctic/pole/dateline domains.

    Important:
    ----------
    The polygon is created in a projected Arctic CRS, not in EPSG:4326.
    That avoids antimeridian and pole-topology problems.

    Parameters
    ----------
    grid
        Grid object understood by lat_lon_from_grid(grid).
    out_shapefile : str or Path
        Output .shp path.
    out_crs : CRS or str or None
        Projected CRS for polygon creation. If None, uses Arctic LAEA.
    fix_geometry : bool
        If True, applies buffer(0) to fix small self-intersections.
    return_gdf : bool
        If True, returns the GeoDataFrame.

    Returns
    -------
    shapely geometry or geopandas.GeoDataFrame
    """
    lat2d, lon2d = _grid_lat_lon_2d(grid)
    ring_lonlat = _outer_boundary_ring_from_grid(lat2d, lon2d)

    if out_crs is None:
        out_crs = get_default_arctic_crs()
    out_crs = CRS.from_user_input(out_crs)

    ring_lon = ring_lonlat[:, 0]
    ring_lat = ring_lonlat[:, 1]
    x, y = _project_lonlat_to_xy(ring_lon, ring_lat, out_crs)

    polygon = Polygon(np.column_stack([x, y]))

    if fix_geometry:
        polygon = polygon.buffer(0)

    if polygon.is_empty:
        raise ValueError("Created polygon is empty.")

    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[polygon], crs=out_crs)

    out_shapefile = Path(out_shapefile)
    out_shapefile.parent.mkdir(parents=True, exist_ok=True)

    if out_shapefile.suffix.lower() != ".shp":
        raise ValueError("Output path must end with '.shp'.")

    gdf.to_file(out_shapefile, driver="ESRI Shapefile")

    return gdf if return_gdf else polygon


def _mask_from_single_polygon_xy(
    polygon: Polygon,
    points_xy: np.ndarray,
    *,
    boundary_radius: float = 1e-9,
) -> np.ndarray:
    """
    Rasterize one projected polygon onto projected point locations.
    """
    exterior = np.asarray(polygon.exterior.coords)
    mask = MplPath(exterior[:, :2]).contains_points(points_xy, radius=boundary_radius)

    for interior in polygon.interiors:
        hole = np.asarray(interior.coords)
        hole_mask = MplPath(hole[:, :2]).contains_points(points_xy, radius=boundary_radius)
        mask &= ~hole_mask

    return mask


def _geometry_to_mask_xy(
    geometry,
    x2d: np.ndarray,
    y2d: np.ndarray,
    *,
    boundary_radius: float = 1e-9,
) -> np.ndarray:
    """
    Convert a projected shapely geometry into a mask on projected grid points.
    """
    points_xy = np.column_stack([x2d.ravel(), y2d.ravel()])
    flat_mask = np.zeros(points_xy.shape[0], dtype=bool)

    def _accumulate(geom):
        nonlocal flat_mask

        if geom.is_empty:
            return

        if isinstance(geom, Polygon):
            flat_mask |= _mask_from_single_polygon_xy(
                geom, points_xy, boundary_radius=boundary_radius
            )
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                flat_mask |= _mask_from_single_polygon_xy(
                    poly, points_xy, boundary_radius=boundary_radius
                )
        elif isinstance(geom, GeometryCollection):
            for subgeom in geom.geoms:
                _accumulate(subgeom)
        else:
            return

    _accumulate(geometry)
    return flat_mask.reshape(x2d.shape)


def make_mask_from_shapefile(
    grid,
    mask_shapefile: PathLike,
    *,
    work_crs: CRS | str | None = None,
    inside_value: float = 1.0,
    outside_value: float = 0.0,
    invert: bool = False,
    dtype=np.float32,
    boundary_radius: float = 1e-9,
    shapefile_crs_if_missing: CRS | str | None = None,
) -> np.ndarray:
    """
    Create a gridded mask from a shapefile on the given grid.

    Robust for Arctic domains because masking is done in projected x/y space.

    Parameters
    ----------
    grid
        Grid object understood by lat_lon_from_grid(grid).
    mask_shapefile : str or Path
        Polygon shapefile.
    work_crs : CRS or str or None
        CRS in which masking is performed.
        If None:
          - use shapefile CRS if it is already projected,
          - otherwise use default Arctic LAEA.
    inside_value, outside_value
        Output values of the mask.
    invert : bool
        If True, swap inside and outside.
    dtype
        Output dtype.
    boundary_radius : float
        Slight positive value to include points on the boundary more robustly.
    shapefile_crs_if_missing : CRS or str or None
        Optional CRS to assign if the shapefile has no CRS metadata.

    Returns
    -------
    mask : np.ndarray
        2D grid mask.
    """
    lat2d, lon2d = _grid_lat_lon_2d(grid)

    gdf = gpd.read_file(mask_shapefile)
    if len(gdf) == 0:
        raise ValueError(f"No geometries found in shapefile: {mask_shapefile}")

    if gdf.crs is None:
        if shapefile_crs_if_missing is None:
            raise ValueError(
                f"Shapefile {mask_shapefile} has no CRS. "
                "Pass shapefile_crs_if_missing=... if you know it."
            )
        gdf = gdf.set_crs(shapefile_crs_if_missing)

    if work_crs is None:
        shp_crs = CRS.from_user_input(gdf.crs)
        work_crs = shp_crs if shp_crs.is_projected else get_default_arctic_crs()

    work_crs = CRS.from_user_input(work_crs)
    gdf = gdf.to_crs(work_crs)

    x2d, y2d = _project_lonlat_to_xy(lon2d, lat2d, work_crs)

    geometry = unary_union(gdf.geometry.values)
    inside = _geometry_to_mask_xy(
        geometry,
        x2d,
        y2d,
        boundary_radius=boundary_radius,
    )

    if invert:
        inside = ~inside

    mask = np.where(inside, inside_value, outside_value).astype(dtype)
    return mask


def circular_mean_deg(lon_deg: np.ndarray) -> float:
    """
    Circular mean of longitudes in degrees, robust to antimeridian crossing.
    Returns value in [-180, 180).
    """
    lon_deg = np.asarray(lon_deg, dtype=float)
    lon_deg = lon_deg[np.isfinite(lon_deg)]
    if lon_deg.size == 0:
        return 0.0

    ang = np.deg2rad(lon_deg)
    mean_ang = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))
    lon0 = np.rad2deg(mean_ang)

    # normalize to [-180, 180)
    lon0 = ((lon0 + 180.0) % 360.0) - 180.0
    return float(lon0)


def infer_arctic_work_crs_from_grid(grid) -> CRS:
    """
    Choose an Arctic LAEA CRS centered near the grid's circular-mean longitude.
    This is better than lon_0=0 for grids crossing the antimeridian.
    """
    lat2d, lon2d = _grid_lat_lon_2d(grid)
    lon0 = circular_mean_deg(lon2d)
    return get_default_arctic_crs(lon_0=lon0)


def download_natural_earth_polygon_shapefile(
    out_shapefile: str | Path,
    *,
    kind: Literal["land", "ocean"] = "land",
    out_crs: CRS | str | None = None,
    dissolve: bool = True,
    fix_geometry: bool = True,
) -> Path:
    """
    Download Natural Earth 10m land/ocean polygons, optionally dissolve them,
    reproject, and save as a local shapefile.
    """
    if kind not in NE_URLS:
        raise ValueError(f"Unsupported kind={kind!r}. Expected 'land' or 'ocean'.")

    out_shapefile = Path(out_shapefile)
    if out_shapefile.suffix.lower() != ".shp":
        raise ValueError("out_shapefile must end with '.shp'")

    out_shapefile.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        zip_path = tmpdir / f"ne_{kind}.zip"
        extract_dir = tmpdir / f"ne_{kind}"

        urllib.request.urlretrieve(NE_URLS[kind], zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        shp_files = list(extract_dir.glob("*.shp"))
        if not shp_files:
            raise FileNotFoundError(f"No .shp found in downloaded archive for {kind}")

        gdf = gpd.read_file(shp_files[0])

        if dissolve:
            geom = unary_union(gdf.geometry.values)
            if fix_geometry:
                geom = geom.buffer(0)
            gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs=gdf.crs)
        elif fix_geometry:
            gdf["geometry"] = gdf.geometry.buffer(0)

        if out_crs is not None:
            out_crs = CRS.from_user_input(out_crs)
            gdf = gdf.to_crs(out_crs)

        gdf.to_file(out_shapefile, driver="ESRI Shapefile")

    return out_shapefile


def make_natural_earth_mask_for_grid(
    grid,
    out_shapefile: str | Path,
    *,
    kind: Literal["land", "ocean"] = "land",
    work_crs: CRS | str | None = None,
    inside_value: float = 1.0,
    outside_value: float = 0.0,
    invert: bool = False,
    dtype=np.float32,
    boundary_radius: float = 1e-9,
) -> np.ndarray:
    """
    Full workflow:
      1) infer a good Arctic projected CRS from the grid,
      2) download and save Natural Earth shapefile in that CRS,
      3) apply mask to the grid.

    Returns
    -------
    mask : np.ndarray
        2D mask on the grid.
    """
    if work_crs is None:
        work_crs = infer_arctic_work_crs_from_grid(grid)

    shp_path = download_natural_earth_polygon_shapefile(
        out_shapefile,
        kind=kind,
        out_crs=work_crs,
        dissolve=True,
        fix_geometry=True,
    )

    mask = make_mask_from_shapefile(
        grid,
        shp_path,
        work_crs=work_crs,
        inside_value=inside_value,
        outside_value=outside_value,
        invert=invert,
        dtype=dtype,
        boundary_radius=boundary_radius,
    )
    return mask