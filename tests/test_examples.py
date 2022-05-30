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


named_examples = examples.available()


class Test_run:
    @pytest.mark.parametrize("named_example", named_examples)
    def test_matches_output(self, named_example):
        result, expected = examples.run(named_example)
        assert result == expected
