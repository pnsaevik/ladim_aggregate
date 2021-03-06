SCRIPT_NAME = "ladim_aggregate"


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
    # '  grid_2D:  Basic example summing up particles in a two-dimensional grid\n'
    # '  grid_3D:  Shows different ways of specifying grid bins\n'
    # '  time:     Make one aggregation for every time step\n'
    # '  filter:   Filter out particles prior to aggregation\n'
    # '  weights:  Use a weighting variable (or expression)\n'
    # '  wgt_tab:  Use an external table to assign weights\n'
    # '  last:     Use only last particle position\n'
    # '  groupby:  Group by specific attribute, such as farm id etc.\n'
    # '  multi:    Data is spread across multiple input and output files\n'
    # '  blur:     Apply a blurring filter the output grid\n'
    # '  crs:      Use a georeferenced output grid\n'
    # '  density:  Divide by volume (or area)\n'
    # '  geotag:   Assign geographic region based on location\n'
    # '  connect:  Group by start and stop region\n'

    example_list = []
    for name in example_names:
        ex = Example(name)
        example_list.append(f'  {name:8}  {ex.descr}')

    parser = argparse.ArgumentParser(
        prog='ladim_aggregate',
        description=(
            "Aggregate particles from LADiM simulations\n\n"
        ),
        epilog=(
            'The program includes several built-in examples:\n'
            + "\n".join(example_list) +
            '\n\nUse "ladim_aggregate --example name_of_example" to run any of these.\n'
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
    logger = logging.getLogger(__name__)
    logger.info(f'Starting ladim_aggregate, version {version_str}')

    # Extract example if requested
    if parsed_args.example:
        ex = Example(config_file)
        config_file = ex.extract()

    import yaml
    logger.info(f'Open config file "{config_file}"')
    with open(config_file, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger.info(f'Input file pattern: "{config["infile"]}"')
    from .input import LadimInputStream
    dset_in = LadimInputStream(config['infile'])
    logger.info(f'Number of input datasets: {len(dset_in.datasets)}')

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
        dset_in.add_grid_variable(data_array=gridvar_spec['data'])

    # Add weights
    if 'weights' in config:
        dset_in.add_derived_variable(varname='weights', definition=config['weights'])

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
        varname='histogram',
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
    for chunk_in in dset_in.chunks(filters=filter_spec):
        if chunk_in.dims['pid'] == 0:
            continue

        # Write histogram values to file
        for chunk_out in hist.make(chunk_in):
            txt = ", ".join([f'{a.start}:{a.stop}' for a in chunk_out['indices']])
            logger.info(f'Write output chunk [{txt}]')
            dset_out.incrementData(
                varname='histogram',
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
    package_logger.setLevel(loglevel)
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    formatter = logging.Formatter('%(asctime)s  %(name)s:%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    package_logger.addHandler(ch)
