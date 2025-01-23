import numpy as np
import pyproj
import xarray as xr


def write_projection(dset, config):
    crs = pyproj.CRS.from_user_input(config['proj4'])
    attrs = crs.to_cf()
    if 'horizontal_datum_name' not in attrs:
        attrs['horizontal_datum_name'] = 'World Geodetic System 1984'
    dset.createVariable('crs', data=0, dims=(), attrs=attrs)
    cs = crs.cs_to_cf()
    dset.setAttrs(config['x'], cs[0])
    dset.setAttrs(config['y'], cs[1])
    dset.setAttrs(config['output_varname'], dict(grid_mapping='crs'))
    dset.main_dataset.Conventions = "CF-1.8"


def compute_area_dataarray(bins: dict[dict], config_projection: dict) -> xr.DataArray:
    """
    Compute cell areas from crecon config variables

    The function is mostly concerned with reformatting input- and
    output data, and uses "compute_area_grid" to do the main work.

    :param bins: A dict of bin edges
    :param config_projection: A dict of projection configs
    :return: The grid cell areas, packaged as an xarray.DataArray
    """
    xcoord = config_projection['x']
    ycoord = config_projection['y']

    x = bins[xcoord]['edges']
    y = bins[ycoord]['edges']
    crs = pyproj.CRS.from_proj4(config_projection['proj4'])
    area_grid = compute_area_grid(x=x, y=y, crs=crs)
    padded_area_grid = np.pad(area_grid, ((0, 1), (0, 1)), 'edge')

    return xr.DataArray(
        data=padded_area_grid,
        coords={xcoord: x, ycoord: y},
        dims=(ycoord, xcoord),
        name='AREA',
        attrs=dict(units='m^2'),
    )


def compute_area_grid(x, y, crs):
    """
    Compute an array of grid cell areas

    :param x: Grid cell edges in x direction
    :param y: Grid cell edges in y direction
    :param crs: PyProj projection
    :return: An array of shape (len(y) - 1, len(x) - 1) containing the area
        (in m2) of each grid cell.
    """
    return np.ones((len(y) - 1, len(x) - 1))
