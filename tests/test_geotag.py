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


class Test_lookup:
    def test_correct_when_1d_indicator_array(self):
        indicator = [False, True, False]
        values = ['A', 'B', 'C']
        result = geotag.lookup(indicator, values, None)
        assert result == 'B'

    def test_returns_missing_value_when_no_indicator(self):
        indicator = [False, False, False]
        values = ['A', 'B', 'C']
        result = geotag.lookup(indicator, values, 'D')
        assert result == 'D'

    def test_correct_when_2d_indicator_array(self):
        indicator = [
            [False, True],
            [False, False],
            [True, False],
        ]
        values = ['A', 'B', 'C']
        result = geotag.lookup(indicator, values, None)
        assert result.tolist() == ['C', 'A']


class Test_get_tag:
    def test_correct_when_standard_input(self):
        result = geotag.get_tag(
            xy=[(1, 1), (3, 3), (5, 5)],
            polygons=[
                geotag.Polygon(shell=[(0, 0), (2, 0), (2, 2), (0, 2)], holes=[]),
                geotag.Polygon(shell=[(2, 2), (4, 2), (4, 4), (2, 4)], holes=[]),
            ],
            attributes=[101, 202],
            missing=404,
        )
        assert list(result) == [101, 202, 404]


class Test_polygons_contains_points:
    def test_correct_when_simple_polygon(self):
        poly = geotag.Polygon(shell=[(10, 0), (12, 0), (12, 2), (10, 2)], holes=[])
        xy = [(11, 1), (13, 3)]
        result = poly.contains_points(xy)
        assert list(result) == [True, False]

    def test_returns_boolean_value_when_boundary(self):
        poly = geotag.Polygon(shell=[(0, 0), (2, 0), (2, 2), (0, 2)], holes=[])
        xy = [(0, 0), (2, 2), (1, 2), (0, 1)]
        result = poly.contains_points(xy)
        assert set(result) - {True, False} == set()

    def test_correct_when_holes(self):
        poly = geotag.Polygon(
            shell=[(10, 0), (15, 0), (15, 5), (10, 5)],
            holes=[
                [(11, 1), (13, 1), (13, 3), (11, 3)]
            ],
        )
        xy = [(12, 2), (14, 4), (16, 6)]
        result = poly.contains_points(xy)
        assert list(result) == [False, True, False]


class Test_Polygon_from_geojson_feature:
    def test_correct_when_no_holes(self):
        coords = [[[0, 0], [2, 0], [2, 2], [0, 2]]]
        f = {"geometry": {"coordinates": coords}}
        poly = geotag.Polygon.from_geojson_feature(f)
        assert poly.shell.vertices.tolist() == coords[0]
        assert len(poly.holes) == 0

    def test_correct_when_holes(self):
        shell = [[0, 0], [3, 0], [3, 3], [0, 3]]
        hole = [[1, 1], [2, 1], [2, 2]]
        coords = [shell, hole]
        f = {"geometry": {"coordinates": coords}}
        poly = geotag.Polygon.from_geojson_feature(f)
        assert poly.shell.vertices.tolist() == shell
        assert len(poly.holes) == 1
        assert poly.holes[0].vertices.tolist() == hole
