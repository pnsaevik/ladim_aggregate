import netCDF4
import xarray as xr
from ladim_aggregate import output
import pytest
import numpy as np
from uuid import uuid4


class Test_MultiDataset:
    @pytest.fixture(scope="function")
    def mdset(self):
        with output.MultiDataset(uuid4(), diskless=True) as d:
            yield d

    def test_can_set_main_coordinate(self, mdset):
        mdset.createCoord('mycoord', [1, 2, 3])
        assert mdset.getCoord('mycoord')[:].tolist() == [1, 2, 3]

    def test_can_set_main_variable(self, mdset):
        mdset.createCoord('A', [1, 2, 3])
        mdset.createCoord('B', [10, 20])
        mdset.createVariable('myvar', np.arange(6).reshape((2, 3)), ('B', 'A'))
        assert mdset.getData('myvar').tolist() == [[0, 1, 2], [3, 4, 5]]

    def test_can_get_partial_data_of_main_variable(self, mdset):
        mdset.createCoord('A', [1, 2, 3])
        mdset.createCoord('B', [10, 20])
        mdset.createVariable('myvar', np.arange(6).reshape((2, 3)), ('B', 'A'))

        data = mdset.getData('myvar')
        idx = np.s_[:1, 1:]
        assert mdset.getData('myvar', idx).tolist() == data[idx].tolist()

    def test_can_set_attrs_when_creating_main_coord_or_variable(self, mdset):
        mdset.createCoord('mycoord', [1, 2, 3], attrs=dict(myatt=123))
        mdset.createVariable('myvar', [4, 5, 6], 'mycoord', attrs=dict(myatt=9))
        assert mdset.getAttrs('mycoord') == dict(myatt=123)
        assert mdset.getAttrs('myvar') == dict(myatt=9)

    def test_can_add_attrs_after_creating_main_coord_or_variable(self, mdset):
        mdset.createCoord('mycoord', [1, 2, 3], attrs=dict(myatt=123))
        mdset.createVariable('myvar', [4, 5, 6], 'mycoord', attrs=dict(myatt=9))
        mdset.setAttrs('mycoord', dict(myatt2=234))
        mdset.setAttrs('myvar', dict(myatt2=8))
        assert mdset.getAttrs('mycoord') == dict(myatt=123, myatt2=234)
        assert mdset.getAttrs('myvar') == dict(myatt=9, myatt2=8)

    def test_can_set_data_of_main_variable_after_creation(self, mdset):
        mdset.createCoord('mycoord', [1, 2, 3])
        mdset.createVariable('myvar', [4, 5, 6], 'mycoord')
        mdset.setData('myvar', [7, 8, 9])
        assert mdset.getData('myvar').tolist() == [7, 8, 9]

    def test_can_set_partial_data_of_main_variable_after_creation(self, mdset):
        mdset.createCoord('mycoord', [1, 2, 3])
        mdset.createVariable('myvar', [4, 5, 6], 'mycoord')
        mdset.setData('myvar', 7, idx=0)
        assert mdset.getData('myvar').tolist() == [7, 5, 6]
        mdset.setData('myvar', [0, 9], idx=np.s_[1:])
        assert mdset.getData('myvar').tolist() == [7, 0, 9]

    def test_can_set_initial_constant_data_of_multifile_variable_at_creation(self, mdset):
        mdset.createCoord('A', [1, 2, 3])
        mdset.createCoord('B', [4, 5], cross_dataset=True)

        mdset.createVariable('x', 1, ('B', 'A'))
        assert mdset.getData('x').tolist() == [[1, 1, 1], [1, 1, 1]]

    def test_can_set_attrs_of_multifile_variable(self, mdset):
        mdset.createCoord('A', [1, 2, 3])
        mdset.createCoord('B', [4, 5], cross_dataset=True)
        mdset.createVariable('x', 1, ('B', 'A'), attrs=dict(myatt=123))
        assert mdset.getAttrs('x') == dict(myatt=123)

    def test_can_set_initial_slicewise_data_of_multifile_variable_at_creation(self, mdset):
        mdset.createCoord('A', [1, 2, 3])
        mdset.createCoord('B', [4, 5], cross_dataset=True)

        mdset.createVariable('x', np.arange(3).reshape((1, 3)), ('B', 'A'))
        assert mdset.getData('x').tolist() == [[0, 1, 2], [0, 1, 2]]

    def test_can_get_partial_data_of_multifile_variable(self, mdset):
        mdset.createCoord('A', [1, 2, 3, 4])
        mdset.createCoord('B', [5, 6, 7, 8, 9], cross_dataset=True)

        mdset.createVariable('x', np.arange(4), ('B', 'A'))
        assert mdset.getData('x', np.s_[1:3, 1:4]).tolist() == [[1, 2, 3], [1, 2, 3]]

    def test_can_set_partial_data_of_multifile_variable_after_creation(self, mdset):
        mdset.createCoord('A', [1, 2, 3])
        mdset.createCoord('B', [5, 6, 7, 8], cross_dataset=True)

        mdset.createVariable('x', 1, ('B', 'A'))
        mdset.setData('x', [[8], [9]], np.s_[1:3, 1:2])
        data = mdset.getData('x').tolist()
        assert data == [[1, 1, 1], [1, 8, 1], [1, 9, 1], [1, 1, 1]]

    def test_can_set_partial_data_when_multiple_crosscoords(self, mdset):
        mdset.createCoord('A', [1, 2])
        mdset.createCoord('B', [3, 4, 5], cross_dataset=True)
        mdset.createCoord('C', [6, 7, 8, 9])
        mdset.createCoord('D', [8, 7, 6, 5, 4], cross_dataset=True)

        mdset.createVariable('x', 1, ('A', ))
        mdset.createVariable('y', 2, ('B', 'A', ))
        mdset.createVariable('z', 3, ('C', 'B', 'A', ))
        mdset.createVariable('w', 4, ('D', 'C', 'B', 'A', ))

        in_data = np.arange(6).reshape((1, 2, 3, 1))
        idx = np.s_[1::5, :2, -3:, 0:1]
        mdset.setData('w', in_data, idx)
        out_data = mdset.getData('w')
        assert out_data[idx].ravel().tolist() == in_data.ravel().tolist()

    def test_cannot_create_variables_after_subdataset_creation(self, mdset):
        # Create main-dataset coordinate (does not lock dataset)
        mdset.createCoord('A', [1, 2])
        # Create main-dataset variable (does not lock datset)
        mdset.createVariable('x', 1, 'A')
        # Set main-dataset variable data (does not lock dataset)
        mdset.setData('x', 2)
        # Create cross-dataset coordinate (does not lock dataset)
        mdset.createCoord('B', [1, 2, 3], cross_dataset=True)
        # Create cross-dataset variable (does not lock dataset)
        mdset.createVariable('y', 3, 'B')
        # Access cross-dataset variable (DOES lock dataset)
        mdset.setData('y', 4)
        # Cannot create new main-dataset coordinates when locked
        with pytest.raises(TypeError):
            mdset.createCoord('C', [1, 2])
        # Cannot create new cross-dataset coordinates when locked
        with pytest.raises(TypeError):
            mdset.createCoord('D', [1, 2], cross_dataset=True)
        # Cannot create new main-dataset variables when locked
        with pytest.raises(TypeError):
            mdset.createVariable('z', 5, 'A')
        # Cannot create new cross-dataset variables when locked
        with pytest.raises(TypeError):
            mdset.createVariable('w', 6, 'B')
        # CAN access variable data after locking
        mdset.setData('x', 7)
        assert mdset.getData('x').tolist() == [7, 7]
        mdset.setData('y', 8)
        assert mdset.getData('y').tolist() == [8, 8, 8]

    def test_cannot_set_crossvar_attributes_after_subdataset_creation(self, mdset):
        # Create main-dataset coordinate (does not lock dataset)
        mdset.createCoord('main_coord', [1, 2])
        # Set main-dataset coordinate attributes (does not lock dataset)
        mdset.setAttrs('main_coord', dict(main_coord_att=2))
        # Create main-dataset variable (does not lock datset)
        mdset.createVariable('main_var', 1, 'main_coord')
        # Set main-dataset variable attributes (does not lock dataset)
        mdset.setAttrs('main_var', dict(main_var_att=3))
        # Create cross-dataset coordinate (does not lock dataset)
        mdset.createCoord('cross_coord', [1, 2, 3], cross_dataset=True)
        # Set cross-dataset coordinate attributes (does not lock dataset)
        mdset.setAttrs('cross_coord', dict(cross_coord_att=4))
        # Create cross-dataset variable (does not lock dataset)
        mdset.createVariable('cross_var', 3, 'cross_coord')
        # Set cross-dataset variable attributes (does not lock dataset)
        mdset.setAttrs('cross_var', dict(cross_var_att=5))
        # Access cross-dataset variable (DOES lock dataset)
        mdset.setData('cross_var', 4)
        # Cannot edit any attributes when locked
        with pytest.raises(TypeError):
            mdset.setAttrs('main_coord', dict(newatt=3))
        with pytest.raises(TypeError):
            mdset.setAttrs('main_var', dict(newatt=3))
        with pytest.raises(TypeError):
            mdset.setAttrs('cross_var', dict(newatt=3))
        with pytest.raises(TypeError):
            mdset.setAttrs('cross_coord', dict(newatt=3))

    def test_accepts_in_memory_datasets(self):
        import netCDF4 as nc
        with nc.Dataset(uuid4(), mode='w', diskless=True) as dset:
            mdset = output.MultiDataset(dset)

            mdset.createCoord('main_coord', [1, 2])
            assert dset.variables['main_coord'][:].tolist() == [1, 2]

            mdset.createVariable('main_var', 1, 'main_coord')
            assert dset.variables['main_var'][:].tolist() == [1, 1]

            mdset.setData('main_var', 2, idx=[0])
            assert dset.variables['main_var'][:].tolist() == [2, 1]

            # Can read after calling close method
            # (Since underlying dataset is not closed)
            mdset.close()
            assert dset.variables['main_var'][:].tolist() == [2, 1]


class Test_nc_to_dict:
    def test_returns_valid_xarray_dict_representation(self):
        with netCDF4.Dataset(uuid4(), 'w', diskless=True) as dset:
            dset.createDimension('mydim1', 2)
            dset.createDimension('mydim2', 3)
            dset.createVariable('myvar1', int, 'mydim1')[:] = 0
            dset.createVariable('myvar2', int, ('mydim1', 'mydim2'))[:] = 1
            dset['myvar1'].long_name = "Variable 2"
            d = output.nc_to_dict(dset)

        xr_dset = xr.Dataset.from_dict(d)
        assert list(xr_dset.variables) == ['myvar1', 'myvar2']
        assert xr_dset.myvar2.values.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert xr_dset.myvar1.attrs['long_name'] == 'Variable 2'
