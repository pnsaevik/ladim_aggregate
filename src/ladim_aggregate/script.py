SCRIPT_NAME = "crecon"


def main_from_command_line():
    """
    Main CLI entry point, takes input arguments from command line
    """
    import sys
    main(*sys.argv[1:])


def main(*args):
    parsed_args = parse_args(list(args))

    # If error or help message, print and exit
    if isinstance(parsed_args, str):
        print(parsed_args)
        return

    # Otherwise, this is a dict
    assert isinstance(parsed_args, dict)

    init_logger()

    # Extract example if requested
    if parsed_args['example']:
        from .examples import Example
        example_name = parsed_args['config_file']
        ex = Example(example_name)
        config_file = ex.extract()
    else:
        config_file = parsed_args['config_file']

    # Run program
    config = load_config(config_file)
    run_conf(config)


def load_config(file):
    import yaml

    if hasattr(file, 'read'):
        return yaml.safe_load(file)

    assert isinstance(file, str)

    import logging
    logger = logging.getLogger(__name__)
    logger.info(f'Open config file "{file}"')
    with open(file, encoding='utf-8') as fp:
        return yaml.safe_load(fp)


def parse_args(args: list[str]):
    """
    Reformat a list of string arguments into a valid parameter set

    If the input list of parameters is valid, the function returns a dict
    of valid parameter values. Otherwise, the function returns a string
    containing a help text explaining the proper use of the function.

    The return dict is simply {"example": bool, "config_file": str}

    :param args: A list of string arguments
    :return: Either a dict of valid parameter values, or a help text string
    """
    import argparse
    from . import __version__ as version_str

    example_help_text = "\n".join(
        [f'  {k:8}  {v}' for k, v in example_descriptions().items()]
    )

    parser = argparse.ArgumentParser(
        prog='crecon',
        description=(
            f"CRECON - CREate CONcentration files (v. {version_str})\n\n"
            "This script converts LADiM particle files to netCDF\n"
            "concentration files.\n\n"
        ),
        epilog=(
            'The program includes several built-in examples:\n'
            f'{example_help_text}\n\n'
            'Use "crecon --example name_of_example" to run any of these.\n'
            'Example files and output files are extracted to the current directory.\n'
        ),
        exit_on_error=False,
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

    # If called with too few arguments, return help string
    if (len(args) < 1) or ('--help' in args) or ('-h' in args):
        return parser.format_help()

    # If unrecognized arguments, return help string
    # This part is unnecessary for modern python versions (issue #121018 in python repo)
    _, unknown_args = parser.parse_known_args(args)
    if unknown_args:
        return f"ERROR: Unknown arguments {unknown_args}\n\n" + parser.format_usage()

    try:
        parsed_args = parser.parse_args(args)

    # If any parsing errors, capture the error and return a string
    except argparse.ArgumentError as e:
        return f"ERROR: {e.message}\n\n" + parser.format_usage()

    # If no errors, return a dict
    return dict(
        example=parsed_args.example,
        config_file=parsed_args.config_file,
    )


def example_descriptions() -> dict[str, str]:
    """
    Returns a sorted dict of available examples with descriptions
    """
    from .examples import Example
    available = Example.available()
    sort_order = [
        'grid_2D', 'grid_3D', 'time', 'filter', 'weights', 'wgt_tab', 'last',
        'groupby', 'multi', 'blur', 'crs', 'density', 'geotag', 'connect',
    ]
    example_names = [n for n in sort_order if n in available]
    example_names += [n for n in available if n not in example_names]

    # Planned for the future:
    # '  blur:     Apply a blurring filter the output grid\n'

    return {k: Example(k).descr for k in example_names}


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
    from . import histogram
    from .parseconfig import parse_config, load_config
    from .proj import compute_area_dataarray, write_projection
    from .input import LadimInputStream
    from .output import MultiDataset
    import numpy as np
    import xarray as xr
    import pandas as pd

    assert isinstance(dset_in, LadimInputStream)
    assert isinstance(dset_out, MultiDataset)

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

    # Add special variable TIMESTEPS
    dset_in.add_derived_variable(
        varname='TIMESTEPS', definition=len(dset_in.timesteps))

    # Prepare histogram bins
    bins = histogram.autobins(config['bins'], dset_in)

    # Add AREA variable
    if 'projection' in config:
        area_dataarray = compute_area_dataarray(bins, config['projection'])
        dset_in.add_grid_variable(data_array=area_dataarray, method="bin")

    # Add weights
    dset_in.add_derived_variable(varname='_auto_weights', definition=config.get('weights', '1'))

    # Add bin indices
    for k, v in bins.items():
        darr = xr.DataArray(data=v['edges'], dims=k, name=f'_BIN_{k}')
        dset_in.add_grid_variable(
            data_array=darr,
            method='bin_idx',
        )

    # Create output coordinate variables
    for coord_name, coord_info in bins.items():
        dset_out.createCoord(
            varname=coord_name,
            data=coord_info['centers'],
            attrs=dset_in.attributes.get(coord_name, {}),
            cross_dataset=coord_name in filesplit_dims,
        )

    # Create aggregation variable
    hist_dtype = np.float32 if 'weights' in config else np.int32
    dset_out.createVariable(
        varname=config['output_varname'],
        data=np.array(0, dtype=hist_dtype),
        dims=tuple(bins.keys()),
    )

    # Add projection information
    if 'projection' in config:
        write_projection(dset_out, config['projection'])

    import logging
    logger = logging.getLogger(__name__)

    # Aggregation algorithm:
    # 1. Read single chunk from ladim file
    # 2. Compute derived variables, including bin idx and weight
    # 3. Group by bin idx and compute weight sum
    # 4. Append result to output dataframe
    # 5. (Persist output dataframe to disk if necessary)
    #

    # Read ladim file timestep by timestep
    dset_in_iterator = dset_in.chunks(
        filters=filter_spec,
        timestep_filter=tsfilter_spec,
        particle_filter=pfilter_spec,
    )

    out_chunk_iterator = histogram.sparse_histogram_chunks_from_dataset_iterator(
        dset_in_iterator,
        bin_cols=[f'_BIN_{k}' for k in bins],
        weight_col='_auto_weights',
        )
    out_chunks = list(out_chunk_iterator)


    # Aggregate output chunks
    df_out = pd.concat(out_chunks, ignore_index=True)
    agg_coords = df_out.iloc[:, :-1].to_numpy().T
    agg_weights = df_out.iloc[:, -1].to_numpy()

    # Write dense output
    out_values, out_indices = histogram.densify_sparse_histogram(agg_coords, agg_weights)
    txt = ", ".join([f'{a.start}:{a.stop}' for a in out_indices])
    logger.debug(f'Write output chunk [{txt}]')
    dset_out.incrementData(
        varname=config['output_varname'],
        data=out_values,
        idx=out_indices,
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

    from . import __version__ as version_str
    package_logger.info(f'Starting CRECON, version {version_str}')


def close_logger():
    import logging
    package_name = str(__name__).split('.', maxsplit=1)[0]
    package_logger = logging.getLogger(package_name)

    # Close the log handlers
    handlers = [h for h in package_logger.handlers]  # Make a copy, otherwise the loop will fail
    for handler in handlers:
        handler.close()
        package_logger.removeHandler(handler)
