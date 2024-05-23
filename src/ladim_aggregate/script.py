SCRIPT_NAME = "crecon"


def main_from_command_line():
    import sys
    main(*sys.argv[1:])


def main(*args):
    import argparse

    from .examples import Example
    available = Example.available()
    sort_order = [
        'grid_2D', 'grid_3D', 'time', 'filter', 'weights', 'wgt_tab', 'last', 'groupby',
        'multi', 'blur', 'crs', 'density', 'geotag', 'connect',
    ]
    example_names = [n for n in sort_order if n in available]
    example_names += [n for n in available if n not in example_names]

    # Planned for the future:
    # '  blur:     Apply a blurring filter the output grid\n'

    example_list = []
    for name in example_names:
        ex = Example(name)
        example_list.append(f'  {name:8}  {ex.descr}')

    from . import __version__ as version_str

    parser = argparse.ArgumentParser(
        prog='crecon',
        description=(
            f"CRECON - CREate CONcentration files (v. {version_str})\n\n"
            "This script converts LADiM particle files to netCDF\n"
            "concentration files.\n\n"
        ),
        epilog=(
            'The program includes several built-in examples:\n'
            + "\n".join(example_list) +
            '\n\nUse "crecon --example name_of_example" to run any of these.\n'
            'Example files and output files are extracted to the current directory.\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        'config_file',
        help="File describing the aggregation options (YAML format)"
    )

    parser.add_argument(
        '--example',
        action='store_true',
        help="Run a built-in example"
    )

    # If called with too few arguments, print usage information
    if len(args) < 1:
        parser.print_help()
        return

    parsed_args = parser.parse_args(args)
    config_file = parsed_args.config_file

    import logging
    from . import __version__ as version_str
    init_logger()

    try:
        logger = logging.getLogger(__name__)
        logger.info(f'Starting CRECON, version {version_str}')

        # Extract example if requested
        if parsed_args.example:
            ex = Example(config_file)
            config_file = ex.extract()

        import yaml
        logger.info(f'Open config file "{config_file}"')
        with open(config_file, encoding='utf-8') as f:
            config = yaml.safe_load(f)

        run_conf(config)

    finally:
        close_logger()


def run_conf(config):
    """
    Run crecon simulation using a configuration dict

    :param config: Configuration
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.debug(f'Input file pattern: "{config["infile"]}"')
    from .input import LadimInputStream

    dset_in = LadimInputStream(config['infile'])
    logger.debug(f'Number of input datasets: {len(dset_in.datasets)}')

    logger.info(f'Create output file "{config["outfile"]}"')
    from .output import MultiDataset

    with MultiDataset(config['outfile']) as dset_out:
        run(dset_in, config, dset_out)


def run(dset_in, config, dset_out, filedata=None):
    from .histogram import Histogrammer, autobins
    from .parseconfig import parse_config, load_config
    import numpy as np

    # Modify configuration dict by reformatting and appending default values
    config = parse_config(config)
    config = load_config(config, filedata)

    # Read some params
    filesplit_dims = config.get('filesplit_dims', [])
    filter_spec = config.get('filter', None)
    tsfilter_spec = config.get('filter_timestep', None)
    pfilter_spec = config.get('filter_particle', None)

    # Add geotagging
    if 'geotag' in config:
        for k in config['geotag']['attrs']:
            spec = ('geotag', dict(
                attribute=k,
                x_var=config['geotag']['coords']['x'],
                y_var=config['geotag']['coords']['y'],
                geojson=config['geotag']['geojson'],
                missing=config['geotag']['outside_value'],
            ))
            dset_in.add_derived_variable(varname=k, definition=spec)

    # Add grid variables
    for gridvar_spec in config.get('grid', []):
        dset_in.add_grid_variable(
            data_array=gridvar_spec['data'],
            method=gridvar_spec['method'],
        )

    # Add derived variables
    for derived_name, derived_spec in config.get('derived', dict()).items():
        dset_in.add_derived_variable(varname=derived_name, definition=derived_spec)

    # Add weights
    if 'weights' in config:
        dset_in.add_derived_variable(varname='_auto_weights', definition=config['weights'])

    # Prepare histogram bins
    bins = autobins(config['bins'], dset_in)
    hist = Histogrammer(bins=bins)
    coords = hist.coords

    # Create output coordinate variables
    for coord_name, coord_info in coords.items():
        dset_out.createCoord(
            varname=coord_name,
            data=coord_info['centers'],
            attrs=coord_info.get('attrs', dict()),
            cross_dataset=coord_name in filesplit_dims,
        )

    # Create aggregation variable
    hist_dtype = np.float32 if 'weights' in config else np.int32
    dset_out.createVariable(
        varname=config['output_varname'],
        data=np.array(0, dtype=hist_dtype),
        dims=tuple(coords.keys()),
    )

    # Add projection information
    if 'projection' in config:
        from .proj import write_projection
        write_projection(dset_out, config['projection'])

    import logging
    logger = logging.getLogger(__name__)

    # Read ladim file timestep by timestep
    dset_in_iterator = dset_in.chunks(
        filters=filter_spec,
        timestep_filter=tsfilter_spec,
        particle_filter=pfilter_spec,
    )

    for chunk_in in dset_in_iterator:
        if chunk_in.sizes['pid'] == 0:
            continue

        # Write histogram values to file
        for chunk_out in hist.make(chunk_in):
            txt = ", ".join([f'{a.start}:{a.stop}' for a in chunk_out['indices']])
            logger.debug(f'Write output chunk [{txt}]')
            dset_out.incrementData(
                varname=config['output_varname'],
                data=chunk_out['values'],
                idx=chunk_out['indices'],
            )

    return dset_out


def init_logger(loglevel=None):
    import logging
    if loglevel is None:
        loglevel = logging.INFO

    package_name = str(__name__).split('.', maxsplit=1)[0]
    package_logger = logging.getLogger(package_name)
    package_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s  %(name)s:%(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    ch.setFormatter(formatter)
    package_logger.addHandler(ch)


def close_logger():
    import logging
    package_name = str(__name__).split('.', maxsplit=1)[0]
    package_logger = logging.getLogger(package_name)

    # Close the log handlers
    handlers = [h for h in package_logger.handlers]  # Make a copy, otherwise the loop will fail
    for handler in handlers:
        handler.close()
        package_logger.removeHandler(handler)
