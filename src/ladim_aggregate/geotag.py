import xarray as xr
from matplotlib.path import Path
import numpy as np


def create_geotagger(attribute, x_var, y_var, geojson, missing_val=np.nan):
    props = np.array([f['properties'][attribute] for f in geojson['features']] + [missing_val])
    coords = [
        np.asarray(f['geometry']['coordinates']).reshape((-1, 2))
        for f in geojson['features']
    ]
    paths = [Path(vertices=c) for c in coords]

    def geotagger(chunk):
        x = chunk[x_var].values
        y = chunk[y_var].values
        xy = np.stack([x, y]).T
        inside = np.asarray([p.contains_points(xy) for p in paths])
        first_nonzero = np.sum(np.cumsum(inside, axis=0) == 0, axis=0)
        return xr.Variable(dims='pid', data=props[first_nonzero])
    return geotagger
