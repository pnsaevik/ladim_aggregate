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
    conf_out['output_varname'] = conf.get('output_varname', 'histogram')
    conf_out['filesplit_dims'] = conf.get('filesplit_dims', [])

    filesplit_bins = {k: "group_by" for k in conf_out['filesplit_dims']}
    conf_out['bins'] = {**filesplit_bins, **conf_out['bins']}

    if 'projection' in conf:
        conf_out['projection']['output_varname'] = conf_out['output_varname']

    for idx, item in enumerate(conf_out['grid']):
        if 'method' not in item:
            conf_out['grid'][idx]['method'] = 'linear'

    return conf_out


def load_config(config, filedata):
    import xarray as xr
    import logging
    logger = logging.getLogger(__name__)

    filedata = filedata or dict()

    # Load geotag file
    if 'geotag' in config:
        fname = config['geotag']['file']
        data = filedata.get(fname, None)
        if data is None:
            logger.info(f'Load geotag file "{fname}"')
            with open(fname, 'br') as f:
                data = f.read()

        import json
        config['geotag']['geojson'] = json.loads(data.decode(encoding='utf-8'))

    # Load grid files
    for grid_spec in config['grid']:
        fname = grid_spec['file']
        if isinstance(fname, xr.Dataset):
            data = grid_spec['file']
        else:
            data = filedata.get(fname, None)  # type: xr.Dataset

        if data is None:
            logger.info(f'Load grid file "{grid_spec["file"]}"')
            with xr.open_dataset(grid_spec['file'], decode_cf=False) as data:
                grid_spec['data'] = data[grid_spec['variable']].copy(deep=True)
        else:
            grid_spec['data'] = data[grid_spec['variable']].copy(deep=True)

    return config
