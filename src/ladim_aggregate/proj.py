def write_projection(dset, config):
    dset.createVariable('crs', 'i2', ())[:] = 0
