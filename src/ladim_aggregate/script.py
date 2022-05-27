def main(*args):
    import argparse

    parser = argparse.ArgumentParser()

    if not args:
        parsed_args = parser.parse_args()
    else:
        parsed_args = parser.parse_args(args)

    config = dict()
    run(**config)


def run(output_file):
    pass
