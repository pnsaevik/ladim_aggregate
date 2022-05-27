import numpy as np


def main(*args):
    import argparse

    parser = argparse.ArgumentParser()

    if not args:
        parsed_args = parser.parse_args()
    else:
        parsed_args = parser.parse_args(args)

    config = dict()
    run(**config)


def run(dset_in, config, dset_out):
    from .histogram import Histogrammer

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
