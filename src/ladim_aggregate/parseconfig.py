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

    for filesplit_dim in conf_out['filesplit_dims']:
        if filesplit_dim not in conf_out['bins'].keys():
            conf_out['bins'][filesplit_dim] = "group_by"

    return conf_out
