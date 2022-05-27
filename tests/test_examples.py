from ladim_aggregate import examples
import netCDF4 as nc
from uuid import uuid4
import pytest
import importlib.resources
import yaml


class Test_nc_dump:
    def test_returns_dict_when_netcdf_input(self):
        with nc.Dataset(uuid4(), 'w', diskless=True) as dset:
            dset = dset  # type: nc.Dataset
            dset.createDimension('time', 3)

            v = dset.createVariable('count', int, 'time')
            v[:] = [4, 2, 2]
            v.long_name = 'Count'

            netcdf_as_dict = examples.nc_dump(dset)

        assert netcdf_as_dict == dict(
            count=dict(dims='time', data=[4, 2, 2], attrs=dict(long_name='Count')),
        )


testfiles = [
    f[:-5]
    for f in importlib.resources.contents('ladim_aggregate.examples')
    if f.endswith('.yaml')
]


class Test_run:
    @pytest.mark.parametrize("testfile", testfiles)
    def test_matches_output(self, testfile):
        spec = importlib.resources.open_text(
            package='ladim_aggregate.examples',
            resource=testfile + '.yaml',
            encoding='utf-8',
        )

        with spec as f:
            testconf = yaml.safe_load(f)

        result = examples.run(testconf)
        assert result == testconf['output_files']
