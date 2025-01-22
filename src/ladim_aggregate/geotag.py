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
    props = attributes_from_geojson(geojson, attribute)
    paths = paths_from_geojson(geojson)

    def geotagger(chunk):
        """
        Returns attributes based on geographical position

        :param chunk: An xarray dataset containing coordinates
        :return: An xarray variable with attributes from enclosing polygons
        """
        x = chunk[x_var].values
        y = chunk[y_var].values
        xy = np.stack([x, y]).T
        point_props = get_tag(xy, paths, props, missing)
        return xr.Variable(dims='pid', data=point_props)

    return geotagger


def attributes_from_geojson(geojson: dict, attribute) -> list:
    """
    Extract attributes from geojson dict

    :param geojson: A geojson dict
    :param attribute: Name of attribute to return
    :return: A list of attribute values
    """
    return [f['properties'][attribute] for f in geojson['features']]


class Polygon:
    def __init__(self, shell, holes):
        holes = holes or []

        self.shell = Path(vertices=shell)
        self.holes = [Path(vertices=h) for h in holes]

    @staticmethod
    def from_geojson_feature(feature):
        coords = feature['geometry']['coordinates']
        return Polygon(shell=coords[0], holes=coords[1:])

    def contains_points(self, xy) -> np.ndarray:
        """
        For each input point, return True (if inside) or False (if not inside)

        :param xy: Input X/Y coordinates
        :return: An array of true/false values
        """
        result = self.shell.contains_points(xy)
        for hole_path in self.holes:
            result = result & (~hole_path.contains_points(xy))
        return result


def paths_from_geojson(geojson: dict) -> list[Polygon]:
    """
    Convert a geojson dict to matplotlib Path objects

    :param geojson: A geojson dict
    :return: A list of matplotlib paths
    """
    features = geojson['features']
    polys = []
    for feature in features:
        polys.append(Polygon.from_geojson_feature(feature))

    return polys


def lookup(indicator, values, missing):
    """
    Use an indicator array to return values from a value array.

    :param indicator: Indicator array (true/false), shape N x M
    :param values: Value array (return first value that corresponds to an
        indicator value of True), shape N
    :param missing: Value to return if entire indicator array is False
    :return: The first corresponding value from the values array, or `missing`
        if the entire indicator value is False, shape M
    """
    np_values = np.asarray(list(values) + [missing])
    first_nonzero = np.sum(np.cumsum(indicator, axis=0) == 0, axis=0)
    return np_values[first_nonzero]


def get_tag(xy, polygons: list[Polygon], attributes: list, missing):
    """
    Return a polygon attribute for each input point

    For each point, find the first polygon that contains the point and
    return the corresponding attribute. If no polygon contains the point,
    return the `missing` value.

    :param xy: Coordinates of points (shape N x 2)
    :param polygons: List of polygons
    :param attributes: One attribute for each polygon
    :param missing: The value to return if some points are not in the polygons
    :return: A list of polygon attributes, one for each point
    """
    inside = np.asarray([p.contains_points(xy) for p in polygons])
    return lookup(inside, attributes, missing)
