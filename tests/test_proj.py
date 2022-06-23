from ladim_aggregate import proj
import netCDF4 as nc
import pytest
from uuid import uuid4


class Test_write_projection:
    @pytest.fixture()
    def nc_dset(self):
        with nc.Dataset(filename=uuid4(), mode='w', diskless=True) as dset:
            yield dset

    def test_adds_crs_variable(self, nc_dset):
        config = dict()
        proj.write_projection(nc_dset, config)
