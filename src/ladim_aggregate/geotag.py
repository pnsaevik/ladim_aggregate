import xarray as xr
from matplotlib.path import Path
import numpy as np


def create_geotagger(attribute, x_var, y_var, geojson, missing=np.nan):
    """
    Create geotagger function.

    :param attribute: Polygon attribute to return
    :param x_var: Name of x coordinate variable
    :param y_var: Name of y coordinate variable
    :param geojson: A dict representation of a geojson file
    :param missing: Value to return if coordinates does not match a polygon
    :return: A geotagger function
    """
    props = np.array([f['properties'][attribute] for f in geojson['features']] + [missing])
    coords = [
        np.asarray(f['geometry']['coordinates']).reshape((-1, 2))
        for f in geojson['features']
    ]
    paths = [Path(vertices=c) for c in coords]

    def geotagger(chunk):
        """
        Returns attributes based on geographical position

        :param chunk: An xarray dataset containing coordinates
        :return: An xarray variable with attributes from enclosing polygons
        """
        x = chunk[x_var].values
        y = chunk[y_var].values
        xy = np.stack([x, y]).T
        inside = np.asarray([p.contains_points(xy) for p in paths])
        first_nonzero = np.sum(np.cumsum(inside, axis=0) == 0, axis=0)
        return xr.Variable(dims='pid', data=props[first_nonzero])
    return geotagger
