# OSTIA sea ice concentration and sea surface temperature

`OstiaSicDataset` and `OstiaSstDataset` share the abstract `OstiaDataset`
reader. Both return one daily field as a float32 NumPy array with axes
`(time, channel, y, x)` and shape `(1, 1, H, W)`.

```python
from datetime import date
from libs.validation.datasets import OstiaSicDataset, OstiaSstDataset

sic = OstiaSicDataset("ostia", name="ostia_sic")
sst_c = OstiaSstDataset("ostia", name="ostia_sst_c")
sst_k = OstiaSstDataset("ostia", name="ostia_sst_k", units="kelvin")

day = date(2023, 1, 1)
concentration = sic[day]   # Percent, 0-100.
temperature_c = sst_c[day] # Degrees Celsius; default units="celsius".
temperature_k = sst_k[day] # Kelvin.
```

The default recursive pattern is
`**/*-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB*.nc`. It accepts year/month folders
and different version suffixes, for example:
`2023/01/20230101120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.1-fv02.1.nc`.
`files_template` can override the pattern; filenames must still start with
the analysis date as `YYYYMMDD`.

Each file must contain one `time` layer on a fixed `latitude`/`longitude`
grid. Coordinates can be 1D axes or matching 2D arrays. Field dimensions
must be `(time, y, x)` in the grid's spatial order. The time coordinate's
calendar date must match the filename; its hour may differ. A missing date
returns `None`. Duplicate files for a date, multiple time layers, changes
in grid coordinates/order, and date mismatches are errors.

The reader uses normal xarray CF decoding for scale, offset and fill values.
SIC uses only `sea_ice_fraction`: decoded fractions outside 0-1 and
non-finite values become NaN, and valid fractions are multiplied by 100.
Zero concentration is preserved. SST uses only `analysed_sst`, whose input
units must be `K` or `kelvin` (case-insensitive). `units="celsius"` subtracts
273.15 before conversion to float32; `units="kelvin"` preserves the decoded
temperature. Negative Celsius temperatures remain valid. Other output-unit
names are rejected. `analysis_error` and the other physical variable are
not loaded or required.

Both readers require `mask`. Land, lakes and rivers (bits 2, 4 and 16) are
excluded. Sea ice (bit 8), including water with sea ice (mask 9), is retained
for both SIC and SST. Missing mask values and values that are not integers
between 1 and 31 become NaN. Masking happens before unit conversion and
spatial interpolation.

Use `dst_grid=model_grid` to apply the existing Dataset interpolation, and
`interpolation_cache_dir="interpolation_weights"` to reuse its weights.
The interpolated result remains float32, with the destination grid's
spatial dimensions. The base interpolation algorithm and its handling of
NaN are unchanged. There is no additional temporal averaging or hourly mode.

Existing model temperature datasets preserve their source-file units.
Before comparing them with OSTIA, check that the model temperature and
the selected OSTIA output units agree.

Product packing, units and mask semantics are described in the
[OSTIA product manual, section V.5](https://documentation.marine.copernicus.eu/PUM/CMEMS-SST-PUM-010-011.pdf).
