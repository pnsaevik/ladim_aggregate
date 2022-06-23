from ladim_aggregate import proj
import netCDF4 as nc
import pytest
from uuid import uuid4


class Test_write_projection:
    @pytest.fixture()
    def nc_dset(self):
        with nc.Dataset(filename=uuid4(), mode='w', diskless=True) as dset:
            dset = dset  # type: nc.Dataset
            dset.createDimension('X', size=4)
            dset.createDimension('Y', size=3)
            dset.createVariable('X', 'f', 'X')[:] = [1, 2, 3, 4]
            dset.createVariable('Y', 'f', 'Y')[:] = [10, 20, 30]
            dset.createVariable('histogram', 'f', ('Y', 'X'))[:] = 0
            yield dset

    def test_adds_crs_variable(self, nc_dset):
        config = dict()
        proj.write_projection(nc_dset, config)
        assert 'crs' in nc_dset.variables
