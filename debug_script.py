import pygrib
filename = '/home/gfs_data/gfs.0p25.2021010100.f006.grib2'
with pygrib.open(filename) as ds:
    print(ds)
    ds.seek(0)
    for grb in ds:
        print(grb)