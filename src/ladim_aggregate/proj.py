import pyproj


def write_projection(dset, config):
    crs = pyproj.CRS.from_user_input(config['proj4'])
    dset.createVariable('crs', data=0, dims=(), attrs=crs.to_cf())
    cs = crs.cs_to_cf()
    dset.setAttrs(config['x'], cs[0])
    dset.setAttrs(config['y'], cs[1])
    dset.setAttrs('histogram', dict(grid_mapping='crs'))
    dset.main_dataset.Conventions = "CF-1.8"
