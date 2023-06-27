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
        config = dict(proj4="+proj=longlat +ellps=WGS84 +datum=WGS84", x='X', y='Y', output_varname='histogram')
        proj.write_projection(nc_dset, config)
        assert nc_dset.getAttrs('crs')['grid_mapping_name'] == 'latitude_longitude'

    def test_adds_attrs_to_coord_vars(self, nc_dset):
        config = dict(proj4="+proj=longlat +ellps=WGS84 +datum=WGS84", x='X', y='Y', output_varname='histogram')
        proj.write_projection(nc_dset, config)
        assert nc_dset.getAttrs('X')['standard_name'] == 'longitude'
        assert nc_dset.getAttrs('Y')['standard_name'] == 'latitude'

    def test_adds_grid_mapping_to_histogram_var(self, nc_dset):
        config = dict(proj4="+proj=longlat +ellps=WGS84 +datum=WGS84", x='X', y='Y', output_varname='histogram')
        proj.write_projection(nc_dset, config)
        assert nc_dset.getAttrs('histogram')['grid_mapping'] == 'crs'

    def test_adds_conventions_attribute_to_dataset(self, nc_dset):
        config = dict(proj4="+proj=longlat +ellps=WGS84 +datum=WGS84", x='X', y='Y', output_varname='histogram')
        proj.write_projection(nc_dset, config)
        assert nc_dset.main_dataset.Conventions == "CF-1.8"
