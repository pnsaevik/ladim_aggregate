from .input import LadimInputStream
from .output import MultiDataset


def main(*args):
    import argparse

    parser = argparse.ArgumentParser(
        prog='ladim_aggregate',
    )

    parser.add_argument('particle_file', help="File containing output of LADiM simulation (netCDF format)")
    parser.add_argument('config_file', help="File describing the aggregation options (YAML format)")

    # If no explicit arguments, use command line arguments
    if not args:
        import sys
        args = sys.argv[1:]

    # If called with no arguments, print usage information
    if len(args) < 2:
        parser.print_help()
        return

    parsed_args = parser.parse_args(args)

    import yaml
    with open(parsed_args.config_file, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    init_logger()

    with LadimInputStream(parsed_args.particle_file) as dset_in:
        with MultiDataset(config['outfile']) as dset_out:
            run(dset_in, config, dset_out)


def run(dset_in, config, dset_out):
    from .histogram import Histogrammer
    import numpy as np

    weights = None
    filesplit_dims = ()

    dset_in.filter = config.get('filter', None)
    dset_in.weights = weights

    # --- Start of section: Find bins

    bins = config['bins']

    autobins = {k: v for k, v in bins.items() if type(v) not in (dict, list)}
    if autobins:
        autolimits = dset_in.find_limits(autobins)
        for k, v in autolimits.items():
            bins[k] = dict(min=v[0], max=v[1], step=bins[k])

    for k, v in bins.items():
        if isinstance(v, dict):
            bins[k] = np.arange(v['min'], v['max'] + v['step'], v['step'])

    # --- End of section: Find bins

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

    for chunk_in in dset_in.chunks():
        for chunk_out in hist.make(chunk_in):
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
