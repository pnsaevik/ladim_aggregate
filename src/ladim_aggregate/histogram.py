import numpy as np
import pandas as pd
import logging
import cftime
import xarray as xr
import datetime


logger = logging.getLogger(__name__)


class Histogrammer:
    def __init__(self, bins=None):
        self.weights = dict(bincount=None)
        self.coords = Histogrammer._get_coords_from_bins(bins)

    @staticmethod
    def _get_coords_from_bins(bins_dict):
        crd = dict()
        for crd_name, bins in bins_dict.items():
            edges = bins['edges']
            centers = bins['centers']
            attrs = bins.get('attrs', dict())
            crd[crd_name] = dict(centers=centers, edges=edges, attrs=attrs)
        return crd

    def make(self, chunk):
        coord_names = list(self.coords.keys())
        bins = [self.coords[k]['edges'] for k in coord_names]
        coords = []
        for k in coord_names:
            logger.debug(f'Load variable "{k}"')
            coords.append(chunk[k].values)

        if '_auto_weights' in chunk.variables:
            weights = chunk['_auto_weights'].values
        else:
            weights = None

        values, idx = adaptive_histogram(coords, bins, weights=weights)
        yield dict(indices=idx, values=values)


def get_centers_from_edges(edges):
    edgediff = edges[1:] - edges[:-1]
    return edges[:-1] + 0.5 * edgediff


def adaptive_histogram(sample, bins, **kwargs):
    """
    Return an adaptive histogram

    For input values `sample` and `bins`, the code snippet

    hist = np.zeros([len(b) - 1 for b in bins])
    hist_chunk, idx = adaptive_histogram(sample, bins, **kwargs)
    hist[idx] = hist_chunk

    gives the same output as

    hist, _ = np.histogramdd(sample, bins, **kwargs)

    :param sample:
    :param bins:
    :param kwargs:
    :return:
    """

    # Cast datetime samples to be comparable with bins
    for i, s in enumerate(sample):
        s_dtype = np.asarray(s).dtype
        if np.issubdtype(s_dtype, np.datetime64):
            sample[i] = s.astype(bins[i].dtype)

    num_entries = next(len(s) for s in sample)
    included = np.ones(num_entries, dtype=bool)

    # Find histogram coordinates of each entry
    binned_sample = []
    for s, b in zip(sample, bins):
        coords = np.searchsorted(b, s, side='right') - 1
        included = included & (0 <= coords) & (coords < len(b) - 1)
        binned_sample.append(coords)

    # Filter out coordinates outside interval
    for i, bs in enumerate(binned_sample):
        binned_sample[i] = bs[included]

    # Abort if there are no points left
    if len(binned_sample[0]) == 0:
        return np.zeros((0, ) * len(sample)), (slice(1, 0), ) * len(sample)

    # Aggregate particles
    df = pd.DataFrame(np.asarray(binned_sample).T)
    df_grouped = df.groupby(list(range(len(bins))))
    if kwargs.get('weights', None) is None:
        df['weights'] = 1
        df_sum = df_grouped.count()
    else:
        df['weights'] = np.asarray(kwargs['weights'])[included]
        df_sum = df_grouped.sum()
    coords = df_sum.index.to_frame().values.T
    vals = df_sum['weights'].values

    # Find min and max bin edges to be used
    idx = [(np.min(c), np.max(c) + 1) for c in coords]
    idx_slice = [slice(start, stop) for start, stop in idx]

    # Densify
    shifted_coords = coords - np.asarray([start for start, _ in idx])[:, np.newaxis]
    shape = [stop - start for start, stop in idx]
    hist_chunk = np.zeros(shape, dtype=vals.dtype)
    hist_chunk[tuple(shifted_coords)] = vals

    return hist_chunk, tuple(idx_slice)


def convert_datebins(spec: dict, dset: xr.Dataset):
    """
    Convert bin specifications containing dates.

    The function takes a bin specification as input, as well as an xarray.Dataset
    where the data variables may potentially contain cfunits date attributes. If one
    of the bin specifications is specified as a date (either date string or python/numpy
    object), and connected to a variable with date attributes, the bin specification is
    converted to a number according to the variable's cfunits date specification.

    The function works for the following types of bin formats:

    1. Pure list - ['2001-01', np.datetime('2001-01-02T12')]
    2. Label format: dict(edges=['2001-01', '2001-02', '2001-03'], labels=['J', 'F'])
    3. Range format: dict(min='2001-01', max='2001-03', step='1 months')

    Date steps can be specified as 'years', 'months', 'days', 'hours', 'minutes',
    'seconds', or a numpy timedelta64 data type.

    :param spec: A dict where the keys are the variable names and the values are the bin
                 specifications.
    :param dset: A dataset where the variables may have cfunits date attributes
    :return:     A dict of the same format as ``spec``, except that dates are converted
                 to numbers.
    """
    newspec = {}

    # k is the variable name
    # v is the bin format
    for k, v in spec.items():
        if k not in dset:
            attrs = {}
        else:
            attrs = dset[k].attrs
        units = attrs.get('units', '')
        calendar = attrs.get('calendar', 'standard')
        newspec[k] = convert_binspec(v, units, calendar)

    return newspec


def convert_binspec(singlespec, units, calendar):
    if 'since' not in units:
        return singlespec

    elif isinstance(singlespec, list):
        return [convert_date(v, units, calendar) for v in singlespec]

    elif isinstance(singlespec, dict):
        result = {k: v for k, v in singlespec.items()}

        if 'min' in result:
            result['min'] = convert_date(result['min'], units, calendar)

        if 'max' in result:
            result['max'] = convert_date(result['max'], units, calendar)

        if 'edges' in result:
            result['edges'] = [convert_date(e, units, calendar) for e in result['edges']]

        if 'step' in result:
            result['step'] = convert_step(result['step'], units, calendar)

        return result

    elif isinstance(singlespec, str) and singlespec.count(' ') == 1:
        return convert_step(singlespec, units, calendar)

    else:
        return singlespec


def convert_step(step_spec, units, calendar):
    if isinstance(step_spec, str):
        step_str, units_in = step_spec.split(sep=' ', maxsplit=1)
        step = float(step_str)
        units_out, _, refdate = units.split(sep=' ', maxsplit=2)
    else:
        return step_spec

    date_in = cftime.num2date(step, f'{units_in} since {refdate}', calendar)
    return cftime.date2num(date_in, f'{units_out} since {refdate}', calendar)


def convert_date(value, units, calendar):
    """
    Convert the input value to a cfunits date format. The input value can be anything
    convertible to np.datetime64.

    :param value: Input date
    :param units: Output cfunits format
    :param calendar: Output cfunits calendar
    :return: Date in cfunits format
    """
    try:
        value = np.datetime64(value).astype(object)
    except ValueError:
        return value

    if isinstance(value, datetime.date):
        value = datetime.datetime.combine(value, datetime.time())

    return cftime.date2num(value, units, calendar)


def autobins(spec, dset):
    if dset is not None and hasattr(dset, 'open_dataset'):
        with dset.open_dataset(0) as xr_dset:
            spec = convert_datebins(spec, xr_dset)

    # Add INIT bins, if any
    for k in spec:
        if k.endswith('_INIT'):
            varname, opname = k.rsplit(sep='_', maxsplit=1)
            dset.add_init_variable(varname)

    # Find bin specification type
    spec_types = dict()
    for k, v in spec.items():
        if isinstance(v, list):
            spec_types[k] = 'edges'

        elif isinstance(v, dict) and all(u in v for u in ['min', 'max', 'step']):
            spec_types[k] = 'range'

        elif isinstance(v, dict) and all(u in v for u in ['edges', 'labels']):
            spec_types[k] = 'edges_labels'

        elif v == 'group_by' or v == 'unique':
            spec_types[k] = 'unique'

        elif np.issubdtype(type(v), np.number):
            spec_types[k] = 'resolution'

        elif isinstance(v, str):
            specsplit = v.split(sep=" ")
            if len(specsplit) == 2:  # Single number with units
                spec_types[k] = 'resolution'
            else:
                raise ValueError(f'Unknown bin type: {v}')

        else:
            raise ValueError(f'Unknown bin type: {v}')

    # Check if we need pre-scanning of the dataset
    scan_params_template = dict(unique=['unique'], resolution=['min', 'max'])
    scan_params = {k: scan_params_template[v] for k, v in spec_types.items()
                   if v in scan_params_template}
    scan_output = {k: dict() for k in spec}
    if scan_params:
        # First add all the variable definitions...
        aggvars = []
        for varname, aggfuncs in scan_params.items():
            for aggfun in aggfuncs:
                if varname.endswith('_INIT'):
                    # If init variable, use the base variable for aggregation
                    aggvar = dset.add_aggregation_variable(varname[:-5], aggfun)
                else:
                    aggvar = dset.add_aggregation_variable(varname, aggfun)
                aggvars.append((aggvar, varname, aggfun))

        logger.info(f'Scan input dataset to find {", ".join([s[0] for s in aggvars])}')

        # ... then trigger the scanning of the dataset
        for aggvar, varname, aggfun in aggvars:
            scan_output[varname][aggfun] = dset.get_aggregation_value(aggvar)

    # Put the specs and the result of the pre-scanning into the bin generator
    bins = {k: bin_generator(spec[k], spec_types[k], scan_output[k]) for k in spec}

    # Add attributes from the dataset
    if dset is not None and hasattr(dset, 'attributes'):
        for k, v in dset.attributes.items():
            if k in bins:
                bins[k]['attrs'] = v

    return bins


def bin_generator(spec, spec_type, scan_output):
    if spec_type == 'edges':
        edges = np.asarray(spec)
        centers = get_centers_from_edges(edges)
    elif spec_type == 'edges_labels':
        edges = np.asarray(spec['edges'])
        centers = np.asarray(spec['labels'])
    elif spec_type == 'range':
        edges = np.arange(spec['min'], spec['max'] + spec['step'], spec['step'])
        centers = get_centers_from_edges(edges)
    elif spec_type == 'unique':
        data = scan_output['unique']
        edges = np.concatenate([data, [data[-1] + 1]])
        centers = np.asarray(data)
    elif spec_type == 'resolution':
        minval = align_to_resolution(scan_output['min'], spec)
        maxval = align_to_resolution(scan_output['max'] + 2 * spec, spec)
        edges = np.arange(minval, maxval, spec)
        centers = get_centers_from_edges(edges)
    else:
        raise ValueError(f'Unknown spec_type: {spec_type}')

    return dict(edges=edges, centers=centers)


def t64conv(timedelta_or_other):
    """
    Convert input data in the form of [value, unit] to timedelta64, or returns the
    argument verbatim if there are any errors. Also accepts string values of the
    form used in cfunits (e.g., "23 hours")
    """
    if isinstance(timedelta_or_other, str) and timedelta_or_other.count(' ') == 1:
        t64val_str, t64unit_cf = timedelta_or_other.split(sep=" ")
        t64val = int(t64val_str)
        if t64unit_cf.endswith('s') and len(t64unit_cf) > 3:  # Remove plurals
            t64unit_cf = t64unit_cf[:-1]
        unitconv = dict(day='D', hour='h', minute='m', second='s', millisecond='ms',
                        microsecond='us')
        t64unit = unitconv.get(t64unit_cf, t64unit_cf)
        return np.timedelta64(t64val, t64unit)

    try:
        t64val, t64unit = timedelta_or_other
        return np.timedelta64(t64val, t64unit)
    except TypeError:
        return timedelta_or_other


# Align to wholenumber resolution
def align_to_resolution(value, resolution):
    """
    Round down to a specified resolution.

    Specifically, `(returned_value <= value) and (returned_value % resolution == 0)`.
    """
    if np.issubdtype(np.array(resolution).dtype, np.timedelta64):
        val_posix = (value - np.datetime64('1970-01-01')).astype('timedelta64[us]')
        res_posix = resolution.astype('timedelta64[us]')
        ret_posix = (val_posix.astype('i8') // res_posix.astype('i8')) * res_posix
        return np.datetime64('1970-01-01') + ret_posix
    else:
        return np.array((value // resolution) * resolution).item()
