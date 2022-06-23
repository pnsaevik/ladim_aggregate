from ladim_aggregate import proj
from ladim_aggregate.output import MultiDataset
import pytest
from uuid import uuid4


class Test_write_projection:
    @pytest.fixture()
    def nc_dset(self):
        with MultiDataset(filename=uuid4(), diskless=True) as dset:
            dset.createCoord('X', data=[1, 2, 3, 4])
            dset.createCoord('Y', data=[10, 20, 30])
            dset.createVariable('histogram', data=0, dims=('Y', 'X'))
            yield dset

    def test_adds_crs_variable(self, nc_dset):
        config = dict()
        proj.write_projection(nc_dset, config)
        assert 'crs' in nc_dset.main_dataset.variables
