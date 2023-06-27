import pyproj


def write_projection(dset, config):
    crs = pyproj.CRS.from_user_input(config['proj4'])
    attrs = crs.to_cf()
    if 'horizontal_datum_name' not in attrs:
        attrs['horizontal_datum_name'] = 'World Geodetic System 1984'
    dset.createVariable('crs', data=0, dims=(), attrs=attrs)
    cs = crs.cs_to_cf()
    dset.setAttrs(config['x'], cs[0])
    dset.setAttrs(config['y'], cs[1])
    dset.setAttrs(config['output_varname'], dict(grid_mapping='crs'))
    dset.main_dataset.Conventions = "CF-1.8"
