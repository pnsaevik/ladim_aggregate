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


def load_yaml_resource(pkg_name, file_name):
    import yaml
    import pkgutil
    pkg_data = pkgutil.get_data(pkg_name, file_name)
    return yaml.safe_load(pkg_data.decode('utf-8'))


def run(example_name):
    # Import resource names
    import importlib
    m = importlib.import_module('.', '.'.join([__name__, example_name]))  # type: Any
    config_file = getattr(m, 'config_file', 'aggregate.yaml')

    # Get all file names
    import importlib.resources
    all_files = [
        f for f in importlib.resources.contents(m.__name__)
        if not f.startswith('__')
    ]

    # Load config file
    config = load_yaml_resource(m.__name__, config_file)

    # Load input datasets (as xarray objects)
    import re
    import xarray as xr
    input_pattern = config['infile'].replace(
        '.nc', '.nc.yaml').replace('.', '\\.').replace('?', '.').replace('*', '.*')
    input_dsets = []
    for input_file in all_files:
        if re.match(input_pattern, input_file):
            xr_dict = load_yaml_resource(m.__name__, input_file)
            input_dsets.append(xr.Dataset.from_dict(xr_dict))

    # Load output datasets (as dict objects)
    output_pattern = config['outfile'].replace('.nc', '.*\\.nc\\.yaml')
    output_dsets = dict()
    for output_file in all_files:
        if re.match(output_pattern, output_file):
            key = output_file[:-5]  # Remove the .yaml part
            output_dsets[key] = load_yaml_resource(m.__name__, output_file)

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
