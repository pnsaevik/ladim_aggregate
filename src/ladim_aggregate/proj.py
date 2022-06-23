import pyproj


def write_projection(dset, config):
    crs = pyproj.CRS.from_proj4(config['proj4'])
    dset.createVariable('crs', data=0, dims=(), attrs=crs.to_cf())
