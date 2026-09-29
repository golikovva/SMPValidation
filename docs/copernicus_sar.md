# Copernicus SAR sea ice concentration

`CopernicusSarSicDataset` reads downloaded `sic_YYYYMMDD.nc` files containing
one time layer on a fixed `latitude`/`longitude` grid. The default recursive
pattern finds files both directly under `path` and in year subdirectories.

```python
from datetime import date
from libs.validation.datasets import CopernicusSarSicDataset

daily = CopernicusSarSicDataset("sar_sic", name="sar_daily")
daily_sic = daily[date(2023, 1, 2)]  # (1, 1, H, W)

hourly = CopernicusSarSicDataset(
    "sar_sic",
    dst_grid=model_grid,
    name="sar_hourly",
    mode="hourly",
    coarsen=(10, 10),
    interpolation_cache_dir="interpolation_weights",
)
hourly_sic = hourly[date(2023, 1, 2)]  # (24, 1, H_model, W_model)
```

The output is a `float32` NumPy array with axes `(time, channel, y, x)` and
concentration in percent (0–100). Missing data, out-of-range SIC, and land
(`status_flag & 1`) become `NaN`; open water remains zero. Other status bits
are not filtered. `standard_deviation_sic` is not read. A missing date returns
`None`; duplicate files for a date or multiple time layers are errors.

Daily mode returns the SIC mosaic without acquisition-time filtering. Hourly
mode places each pixel in the UTC interval `[date + hour, date + hour + 1)`
specified by its `acq_time`. All other pixels, including missing acquisition
times and observations outside the requested date, remain `NaN`. There are
always 24 layers, even for empty hours. This partitions the available mosaic;
it does not reconstruct earlier observations replaced during mosaicking.

Numeric `acq_time` supports CF time units with an epoch and the product's
`units="s"` convention (Unix seconds, zero is missing). Already decoded
datetime values are also supported. Hourly filtering uses the stored times;
it cannot recover precision lost in float32 epoch seconds.

With `dst_grid`, the existing base-class interpolation is applied separately
to each layer after masking, followed only by a float32 conversion. Its
handling of NaN and spatial coverage is unchanged. `files_template` and
`interpolation_cache_dir` use the usual Dataset options. All source files
must have the same coordinates and spatial dimension order.

## Block averaging

`coarsen=10` combines blocks of 10 latitude rows and 10 longitude columns;
`coarsen=(5, 10)` uses separate factors. The default `coarsen=1` (also `(1, 1)`)
preserves the original behavior. Factors must be positive integers.

With coarsening enabled, coordinates must be regular monotonic 1D axes.
Increasing and decreasing axes are supported. Source cell boundaries are
inferred halfway between adjacent centers, with half-step extrapolation at
the edges and latitude boundaries clipped at the poles. SIC is averaged using
spherical cell-area weights, after excluding land and invalid values. The
denominator includes only the area of valid observations; an empty block stays
`NaN`. No minimum coverage threshold is imposed.

Short blocks at the array edges are retained, so the output grid size is
`(ceil(H / latitude_factor), ceil(W / longitude_factor))`. Coarse coordinates
are the midpoints of each block's outer boundaries and do not depend on
missing observations. At least two coarse cells must remain along each axis,
as required by the existing grid's corner estimation.

The original 1D axes are retained to validate subsequent files, but only the
coarse grid is expanded into 2D and passed to `Grid` and the existing
interpolator. Its reconstructed corners remain approximate, particularly
for partial edge blocks; area-weighted averaging does not guarantee exact
conservation by the entire subsequent interpolation pipeline. Coarsening can
also change the interpolator's existing automatic choice of regridding method.

Hourly mode first selects source pixels for a given acquisition hour, then
averages them into that hour's coarse layer. Acquisition times are not averaged,
and the 24 original-resolution hourly layers are never allocated when
coarsening is enabled.

Source daily fields are still read in full, and temporary arrays and base-class
processing require additional memory. Without coarsening, a
`(24, 1, 4917, 30000)` float32 output alone requires approximately 14.2 GB;
with `coarsen=10`, the `(24, 1, 492, 3000)` output is about 142 MB. This reader
does not add spatial subsetting or streaming from disk.
