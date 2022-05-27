import netCDF4 as nc


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
    return dset_out
