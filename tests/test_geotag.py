from ladim_aggregate import geotag
import json
import xarray as xr


class Test_create_geotagger:
    def test_can_return_correct_polygon_attribute_of_particle(self):
        chunk = xr.Dataset(
            data_vars=dict(
                lon=xr.Variable('pid', [.5, .5, 10.5]),
                lat=xr.Variable('pid', [60.5, 70.5, 70.5]),
            )
        )

        geojson = json.loads("""
        {
            "type": "FeatureCollection",
            "name": "layer_name",
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
                }
            },
            "features": [
                {
                    "type": "Feature",
                    "properties": { "region": 101 },
                    "geometry": {
                        "type": "MultiPolygon",
                        "coordinates": [ [ [ [ 0, 60 ], [ 1, 60 ], [ 1, 61 ], [ 0, 61 ], [ 0, 60 ] ] ] ]
                    }
                },
                {
                    "type": "Feature",
                    "properties": { "region": 102 },
                    "geometry": {
                        "type": "MultiPolygon",
                        "coordinates": [ [ [ [ 10, 70 ], [ 11, 70 ], [ 11, 71 ], [ 10, 71 ], [ 10, 70 ] ] ] ]
                    }
                }
            ]
        }
        """)

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
