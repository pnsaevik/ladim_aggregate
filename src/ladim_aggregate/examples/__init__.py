from typing import Any


def nc_dump(dset):
    """Returns the contents of an open netCDF4 dataset as a dict"""

    variables = dict()
    for name in dset.variables:
        v = dict()
        v['dims'] = list(dset.variables[name].dimensions)
        if len(v['dims']) == 1:
            v['dims'] = v['dims'][0]

        v['data'] = dset.variables[name][:].tolist()

        atts = dict()
        for attname in dset.variables[name].ncattrs():
            atts[attname] = dset.variables[name].getncattr(attname)
        if atts:
            v['attrs'] = atts

        variables[name] = v

    return variables


def run(example_name):
    import yaml
    import pkgutil
    pkg = __name__ + '.' + example_name
    files = get_file_list(example_name)

    def load_yaml(fname):
        return yaml.safe_load(pkgutil.get_data(pkg, fname).decode('utf-8'))

    # Load config file
    config = load_yaml(files['config'])
    if 'geotag' in config:
        config['geotag']['geojson'] = load_yaml(config['geotag']['file'])

    # Load input datasets (as xarray objects)
    import xarray as xr
    input_dsets = []
    for input_file in files['input']:
        xr_dict = load_yaml(input_file)
        input_dsets.append(xr.Dataset.from_dict(xr_dict))

    # Load output datasets (as dict objects)
    output_dsets = dict()
    for output_file in files['output']:
        key = output_file[:-5]  # Remove the .yaml part
        output_dsets[key] = load_yaml(output_file)

    from .. import script
    from ..output import MultiDataset
    from ..input import LadimInputStream

    ladim_input_stream = LadimInputStream(input_dsets)

    outfile_name = config['outfile']
    with MultiDataset(outfile_name, diskless=True) as output_dset:
        script.run(ladim_input_stream, config, output_dset)
        result = output_dset.to_dict()

    return result, output_dsets


def available():
    import pkgutil
    return [m.name for m in pkgutil.iter_modules(__path__) if m.ispkg]


def get_descr(example_name):
    # Import resource names
    import importlib
    m = importlib.import_module('.', '.'.join([__name__, example_name]))  # type: Any
    config_file = getattr(m, 'config_file', 'aggregate.yaml')

    import pkgutil
    import re
    config_txt = pkgutil.get_data(m.__name__, config_file).decode('utf-8')
    match = re.match('^# (.*)', config_txt)
    if match:
        return match.group(1)
    else:
        return ""


def get_file_list(example_name):
    # Import resource names
    import importlib
    m = importlib.import_module('.', '.'.join([__name__, example_name]))  # type: Any

    # Get all file names
    import importlib.resources
    all_files = [
        f for f in importlib.resources.contents(m.__name__)
        if not f.startswith('__')
    ]

    import yaml
    import pkgutil
    config_file = getattr(m, 'config_file', 'aggregate.yaml')
    config_str = pkgutil.get_data(m.__name__, config_file).decode('utf-8')
    config = yaml.safe_load(config_str)

    import re
    input_pattern = config['infile'].replace(
        '.nc', '.nc.yaml').replace('.', '\\.').replace('?', '.').replace('*', '.*')
    input_files = [f for f in all_files if re.match(input_pattern, f)]

    output_pattern = config['outfile'].replace('.nc', '.*\\.nc\\.yaml')
    output_files = [f for f in all_files if re.match(output_pattern, f)]

    return dict(all=all_files, config=config_file, input=input_files, output=output_files)


def extract(example_name):
    import yaml
    import pkgutil
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)
    pkg = __name__ + '.' + example_name
    files = get_file_list(example_name)
    outdir = Path('.')

    # Save verbatim files
    for file in files['all']:
        if file.endswith('.geojson') or file == files['config']:
            logger.info(f'Extract file: "{file}"')
            with open(outdir / file, 'bw') as f:
                f.write(pkgutil.get_data(pkg, file))

    # Save input datasets (as xarray objects)
    for input_file in files['input']:
        new_file = input_file[:-5]  # Remove .yaml extension
        logger.info(f'Extract input data file: "{new_file}"')
        yaml_str = pkgutil.get_data(pkg, input_file).decode('utf-8')
        xr_dict = yaml.safe_load(yaml_str)
        import xarray as xr
        xr_dset = xr.Dataset.from_dict(xr_dict)
        xr_dset.to_netcdf(new_file)

    return files['config']
