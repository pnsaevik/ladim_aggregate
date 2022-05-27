from ladim_aggregate import run, input as ladim_input, output as ladim_output
from uuid import uuid4
import pytest
import xarray as xr
import numpy as np


@pytest.fixture(scope='module')
def ladim_dset():
    return xr.Dataset(
        data_vars=dict(
            X=xr.Variable('particle_instance', [5, 5, 6, 6, 5, 6]),
            Y=xr.Variable('particle_instance', [60, 60, 60, 61, 60, 62]),
            Z=xr.Variable('particle_instance', [0, 1, 2, 3, 4, 5]),
            lon=xr.Variable('particle_instance', [5, 5, 6, 6, 5, 6]),
            lat=xr.Variable('particle_instance', [60, 60, 60, 61, 60, 62]),
            instance_offset=xr.Variable((), 0),
            farm_id=xr.Variable('particle', [12345, 12346, 12347, 12348]),
            pid=xr.Variable('particle_instance', [0, 1, 2, 3, 1, 2]),
            particle_count=xr.Variable('time', [4, 2]),
        ),
        coords=dict(
            time=np.array(['2000-01-02', '2000-01-03']).astype('datetime64[D]'),
        ),
    )


class Test_run:
    def test_accepts_minimal_config(self, ladim_dset):
        dset_in = ladim_input.LadimInputStream(ladim_dset)
        dset_out = ladim_output.MultiDataset(uuid4(), diskless=True)
        config = dict()
        run(dset_in, config, dset_out)
