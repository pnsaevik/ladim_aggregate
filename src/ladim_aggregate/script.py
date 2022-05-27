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


def run(output_file, diskless=False):
    from .output import MultiDataset
    MultiDataset(output_file, diskless=diskless)
