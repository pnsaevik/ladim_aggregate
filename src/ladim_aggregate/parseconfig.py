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
    conf_out['filesplit_dims'] = conf.get('filesplit_dims', [])

    filesplit_bins = {k: "group_by" for k in conf_out['filesplit_dims']}
    conf_out['bins'] = {**filesplit_bins, **conf_out['bins']}

    return conf_out
