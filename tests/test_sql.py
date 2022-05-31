from ladim_aggregate import sql
import xarray as xr
import pytest
import numpy as np


@pytest.fixture()
def ladim_dset():
    return xr.Dataset(
        data_vars=dict(
            X=xr.Variable('particle_instance', [1, 1, 3, 3, 5, 5, 7, 7]),
            particle_count=xr.Variable('time', [4, 2, 2]),
            pid=xr.Variable('particle_instance', [0, 1, 2, 3, 1, 2, 1, 2]),
            farm_id=xr.Variable('particle', [12345, 12346, 12345, 12345]),
        ),
        coords=dict(
            time=np.array(
                ['2019-01-02', '2019-01-04', '2019-01-06']).astype('datetime64'),
        )
    )


class Test_add_ladim:
    def test_can_add_ladim_dataset(self, ladim_dset):
        with sql.Particles() as p:
            p.add_ladim(ladim_dset)
