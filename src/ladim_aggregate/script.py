import logging


def main(*args):
    import argparse

    from . import examples
    available = examples.available()
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
        descr = examples.get_descr(name)
        example_list.append(f'  {name:8}  {descr}')

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

    # If no explicit arguments, use command line arguments
    if not args:
        import sys
        args = sys.argv[1:]

    # If called with too few arguments, print usage information
    if len(args) < 2:
        parser.print_help()
        return

    parsed_args = parser.parse_args(args)
    config_file = parsed_args.config_file

    import logging
    init_logger()
    logger = logging.getLogger(__name__)

    # Extract example if requested
    if parsed_args.example:
        from .examples import extract
        config_file = extract(example_name=config_file)

    import yaml
    logger.info(f'Open config file "{config_file}"')
    with open(config_file, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    from .input import LadimInputStream
    logger.info(f'Open ladim file "{config["infile"]}"')
    with LadimInputStream(config['infile']) as dset_in:
        from .output import MultiDataset
        logger.info(f'Create output file "{config["outfile"]}"')
        with MultiDataset(config['outfile']) as dset_out:
            run(dset_in, config, dset_out)


def run(dset_in, config, dset_out):
    from .histogram import Histogrammer, autobins
    from .parseconfig import parse_config
    import numpy as np

    # Modify configuration dict by reformatting and appending default values
    config = parse_config(config)

    filesplit_dims = config.get('filesplit_dims', [])

    dset_in.filter = config.get('filter', None)
    dset_in.weights = config.get('weights', None)

    bins = autobins(config['bins'], dset_in)
    hist = Histogrammer(bins=bins)
    coords = hist.coords

    for coord_name, coord_info in coords.items():
        dset_out.createCoord(
            varname=coord_name,
            data=coord_info['centers'],
            attrs=coord_info.get('attrs', dict()),
            cross_dataset=coord_name in filesplit_dims,
        )

    dset_out.createVariable(
        varname='histogram',
        data=np.array(0, dtype=np.float32),
        dims=tuple(coords.keys()),
    )

    logger = logging.getLogger(__name__)

    for chunk_in in dset_in.chunks():
        for chunk_out in hist.make(chunk_in):
            txt = ", ".join([f'{a.start}:{a.stop}' for a in chunk_out['indices']])
            logger.info(f'Write output chunk [{txt}]')
            dset_out.incrementData(
                varname='histogram',
                data=chunk_out['values'],
                idx=chunk_out['indices'],
            )

    if 'projection' in config:
        from .proj import write_projection
        write_projection(dset_out, config['projection'])

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
