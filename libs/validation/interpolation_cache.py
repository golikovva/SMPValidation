"""Private, optional disk cache for serial ESMPy regridding weights."""

import hashlib
import json
import logging
import os
from pathlib import Path
from uuid import uuid4

import esmpy
import numpy as np


logger = logging.getLogger(__name__)
_CACHE_VERSION = 1
_KEY_ATTRIBUTE = 'smp_interpolation_cache_key'
_VERSION_ATTRIBUTE = 'smp_interpolation_cache_version'


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported cache metadata: {type(value).__name__}')


def _cache_key(src_grid, dst_grid, regrid_options):
    """Hash the actual ordered geometry, topology, options and library versions."""
    constants = esmpy.api.constants
    metadata = {
        'cache_version': _CACHE_VERSION,
        'esmpy_version': str(esmpy.__version__),
        'esmf_version': str(getattr(constants, '_ESMF_VERSION', esmpy.__version__)),
        'field_layout': {'staggerloc': 'CENTER', 'typekind': 'R8'},
        'regrid_options': regrid_options,
    }
    digest = hashlib.sha256()

    def add_metadata(value):
        encoded = json.dumps(
            value, sort_keys=True, separators=(',', ':'), default=_json_default,
        ).encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'little'))
        digest.update(encoded)

    add_metadata(metadata)
    for role, wrapper in (('source', src_grid), ('destination', dst_grid)):
        topology = {
            name: getattr(wrapper.grid, name, None)
            for name in (
                'coord_sys', 'coord_typekind', 'num_peri_dims',
                'periodic_dim', 'pole_dim', 'pole_kind',
            )
        }
        add_metadata({'role': role, 'shape': tuple(wrapper.shape), 'topology': topology})
        for name in ('lat', 'lon', 'lat_corners', 'lon_corners'):
            coordinates = np.ascontiguousarray(getattr(wrapper, name), dtype='<f8')
            add_metadata({'name': name, 'shape': coordinates.shape})
            # Avoid a second full-size bytes copy of large coordinate arrays.
            digest.update(coordinates.reshape(-1).view(np.uint8))
    return digest.hexdigest()


def _has_grid_items(grid):
    """Custom ESMF masks/areas need a separate cache contract, outside v1."""
    for name in ('mask', 'area'):
        items = getattr(grid, name, None)
        if items is None:
            continue
        if isinstance(items, dict):
            items = items.values()
        elif not isinstance(items, (list, tuple)):
            return True
        if any(item is not None for item in items):
            return True
    return False


def _create_regrid(src_grid, dst_grid, src_field, dst_field, options, cache_dir):
    """Return a ready operator; cache I/O failures never disable interpolation."""
    if cache_dir is None:
        return esmpy.Regrid(src_field, dst_field, **options)
    if esmpy.pet_count() != 1:
        logger.warning('Interpolation disk cache is disabled for multi-process ESMF.')
        return esmpy.Regrid(src_field, dst_field, **options)
    if _has_grid_items(src_grid.grid) or _has_grid_items(dst_grid.grid):
        logger.warning('Interpolation disk cache is disabled for custom ESMF masks/areas.')
        return esmpy.Regrid(src_field, dst_field, **options)

    try:
        # Keep this dependency optional when disk caching is not requested.
        from netCDF4 import Dataset as NetCDFDataset

        key = _cache_key(src_grid, dst_grid, options)
        directory = Path(cache_dir).expanduser().resolve()
        path = directory / f'{key}.nc'
    except Exception as exc:
        logger.warning('Cannot prepare interpolation cache %s: %s', cache_dir, exc)
        return esmpy.Regrid(src_field, dst_field, **options)

    try:
        if path.exists():
            with NetCDFDataset(str(path), 'r') as weights:
                if (weights.getncattr(_KEY_ATTRIBUTE) != key
                        or weights.getncattr(_VERSION_ATTRIBUTE) != _CACHE_VERSION):
                    raise ValueError('weight file cache metadata does not match')
            regrid = esmpy.RegridFromFile(src_field, dst_field, filename=str(path))
            logger.info('Loaded interpolation weights from %s', path)
            return regrid
    except Exception as exc:
        logger.warning('Cannot load interpolation cache %s; recalculating: %s', path, exc)

    temporary_path = None
    try:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            # ESMF expects a new filename. Each writer owns just this temporary file.
            temporary_path = directory / f'.{key}.{uuid4().hex}.tmp.nc'
            regrid = esmpy.Regrid(
                src_field, dst_field, filename=str(temporary_path),
                create_rh=True, **options,
            )
        except Exception as exc:
            logger.warning('Cannot write interpolation cache %s; using memory: %s', path, exc)
            # Do not hide errors in the geometry or weight calculation itself.
            return esmpy.Regrid(src_field, dst_field, **options)

        try:
            with NetCDFDataset(str(temporary_path), 'a') as weights:
                weights.setncattr(_KEY_ATTRIBUTE, key)
                weights.setncattr(_VERSION_ATTRIBUTE, _CACHE_VERSION)
            os.replace(temporary_path, path)
            logger.info('Saved interpolation weights to %s', path)
        except Exception as exc:
            # The in-memory operator remains usable even if publication fails.
            logger.warning('Cannot publish interpolation cache %s: %s', path, exc)
        return regrid
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning('Cannot remove temporary weights %s: %s', temporary_path, exc)
