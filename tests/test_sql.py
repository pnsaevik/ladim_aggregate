import pandas as pd
from ladim_aggregate import sql
import xarray as xr
import pytest
import numpy as np
import pandas


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
            part = pd.read_sql('SELECT * FROM particle', p.connection)
            inst = pd.read_sql('SELECT * FROM particle_instance', p.connection)
            time = pd.read_sql('SELECT * FROM time', p.connection)

        assert len(part) == ladim_dset.dims['particle']
        assert len(inst) == ladim_dset.dims['particle_instance']
        assert len(time) == ladim_dset.dims['time']


class Test_get_sql_data:
    def test_correct_when_int(self):
        data = np.array([1, 2, 3]).astype(int)
        dtype, outdata = sql.get_sql_data_format(data)
        assert outdata is data
        assert dtype == 'INTEGER'

    def test_correct_when_float(self):
        data = np.array([1, 2, 3]).astype(float)
        dtype, outdata = sql.get_sql_data_format(data)
        assert outdata is data
        assert dtype == 'REAL'

    def test_correct_when_string(self):
        data = np.array([1, 2, 3]).astype(str)
        dtype, outdata = sql.get_sql_data_format(data)
        assert outdata is data
        assert dtype == 'TEXT'

    def test_correct_when_date(self):
        data = np.array(['2020-01-01', '2020-01-02']).astype('datetime64')
        dtype, outdata = sql.get_sql_data_format(data)
        assert outdata.tolist() == ['2020-01-01T00:00:00', '2020-01-02T00:00:00']
        assert dtype == 'TEXT'
