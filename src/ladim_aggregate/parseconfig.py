def parse_config(conf):
    """
    Parse an input configuration to ladim_aggregate, append default values and make
    formatting changes before it is passed onwards to the program.

    :param conf: Input configuration
    :return: Output configuration, with default values appended and formatting changes
    applied.
    """
    conf_out = conf.copy()

    conf_out['bins'] = conf.get('bins', dict())
    conf_out['grid'] = conf.get('grid', [])
    conf_out['filesplit_dims'] = conf.get('filesplit_dims', [])

    filesplit_bins = {k: "group_by" for k in conf_out['filesplit_dims']}
    conf_out['bins'] = {**filesplit_bins, **conf_out['bins']}

    return conf_out


def load_config(config, filedata):
    import xarray as xr
    filedata = filedata or dict()

    # Load geotag file
    if 'geotag' in config:
        fname = config['geotag']['file']
        data = filedata.get(fname, None)
        if data is None:
            with open(fname, 'br') as f:
                data = f.read()

        import json
        config['geotag']['geojson'] = json.loads(data.decode(encoding='utf-8'))

    # Load grid files
    for grid_spec in config['grid']:
        fname = grid_spec['file']
        data = filedata.get(fname, None)  # type: xr.Dataset
        if data is None:
            with xr.open_dataset(grid_spec['file']) as data:
                grid_spec['data'] = data[grid_spec['variable']].copy(deep=True)
        else:
            grid_spec['data'] = data[grid_spec['variable']].copy(deep=True)

    return config


def load_dataarray(file_name, variable):
    import xarray as xr
    with xr.open_dataset(file_name) as dset:
        v = dset[variable]
        vv = v.copy(deep=True)
        pass

    return vv

