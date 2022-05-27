import contextlib
import numpy as np


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


@contextlib.contextmanager
def nc_load(dset_dict):
    from uuid import uuid4
    import netCDF4 as nc

    dims = {}
    for v in dset_dict.values():
        dimnames = v['dims']
        if isinstance(dimnames, str):
            dimnames = [dimnames]
        for dimname, dimsize in zip(dimnames, np.shape(v['data'])):
            dims[dimname] = dimsize

    with nc.Dataset(uuid4(), 'w', diskless=True) as dset:
        for dimname, dimsize in dims.items():
            dset.createDimension(dimname, dimsize)

        for k, v in dset_dict.items():
            data = np.array(v['data'])
            dset.createVariable(k, data.dtype, v['dims'])[:] = data
            dset.variables[k].setncatts(v.get('attrs', {}))

        yield dset


def run(testconf):
    return testconf['output_files']
