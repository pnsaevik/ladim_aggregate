import xarray as xr


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


def run(testconf):
    from .. import script
    from ..output import MultiDataset
    from ..input import LadimInputStream
    import re

    ladim_filename, conf_filename,  = testconf['command_args']
    conf = testconf['input_files'][conf_filename]
    pattern = ladim_filename.replace('?', '.').replace('*', '.*')

    input_dsets = LadimInputStream([
        xr.Dataset.from_dict(testconf['input_files'][k])
        for k in testconf['input_files']
        if re.match(pattern, k)
    ])

    outfile_name = conf['outfile']
    with MultiDataset(outfile_name, diskless=True) as output_dset:
        script.run(input_dsets, conf, output_dset)
        return output_dset.to_dict()
