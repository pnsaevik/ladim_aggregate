from ladim_aggregate import geotag
import json
import xarray as xr
import pkgutil


class Test_create_geotagger:
    def test_can_return_correct_polygon_attribute_of_particle(self):
        chunk = xr.Dataset(
            data_vars=dict(
                lon=xr.Variable('pid', [.5, .5, 10.5]),
                lat=xr.Variable('pid', [60.5, 70.5, 70.5]),
            )
        )

        pkg = 'ladim_aggregate.examples.connect'
        geojson = json.loads(pkgutil.get_data(pkg, 'regions.geojson').decode('utf-8'))

        geotagger = geotag.create_geotagger(
            attribute="region",
            x_var='lon',
            y_var='lat',
            geojson=geojson,
            missing=-1,
        )

        region = geotagger(chunk)
        assert region.dims == ('pid', )
        assert region.values.tolist() == [101, -1, 102]
