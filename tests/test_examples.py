from ladim_aggregate import examples
import netCDF4 as nc
from uuid import uuid4
import pytest


class Test_Example:
    def test_available(self):
        available = examples.Example.available()
        assert 'grid_2D' in available

    def test_config(self):
        ex = examples.Example('grid_2D')
        assert 'bins' in ex.config

    def test_package(self):
        ex = examples.Example('grid_2D')
        assert ex.package == 'ladim_aggregate.examples.grid_2D'

    def test_descr(self):
        ex = examples.Example('grid_2D')
        assert ex.descr.startswith('Basic example')

    def test_files(self):
        ex = examples.Example('grid_2D')
        data_dict = ex.files()
        assert hasattr(data_dict['aggregate.yaml'], 'decode')  # byte object
        assert hasattr(data_dict['ladim.nc.yaml'], 'decode')  # byte object
        assert hasattr(data_dict['ladim.nc'], 'to_netcdf')  # xarray object

    def test_extract(self, tmp_path):
        ex = examples.Example('grid_2D')
        ex.extract(tmp_path)
        fnames = {f.name for f in tmp_path.glob('*')}
        assert 'ladim.nc' in fnames
        assert 'aggregate.yaml' in fnames
        assert 'count.nc' not in fnames       # Output files not included
        assert 'ladim.nc.yaml' not in fnames  # Only unpacked files are included


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


named_examples = examples.Example.available()


class Test_run:
    @pytest.mark.parametrize("name", named_examples)
    def test_matches_output(self, name):
        ex = examples.Example(name)
        result, expected = ex.run()
        assert result == expected
