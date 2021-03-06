import numpy as np
import pandas as pd
import logging


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
            logger.info(f'Load variable "{k}"')
            coords.append(chunk[k].values)

        if 'weights' in chunk.variables:
            weights = chunk.weights.values
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
        df['weights'] = kwargs['weights']
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


def autobins(spec, dset):
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
        res = t64conv(spec)
        minval = align_to_resolution(scan_output['min'], res)
        maxval = align_to_resolution(scan_output['max'] + 2 * res, res)
        edges = np.arange(minval, maxval, res)
        centers = get_centers_from_edges(edges)
    else:
        raise ValueError(f'Unknown spec_type: {spec_type}')

    return dict(edges=edges, centers=centers)


def t64conv(timedelta_or_other):
    """
    Convert input data in the form of [value, unit] to timedelta64, or returns the
    argument verbatim if there are any errors.
    """
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
