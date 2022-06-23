import numpy as np
import pandas as pd


class Histogrammer:
    def __init__(self, bins=None):
        self.weights = dict(bincount=None)
        self.coords = Histogrammer._get_coords_from_bins(bins)

    @staticmethod
    def _get_coords_from_bins(bins_dict):
        crd = dict()
        for crd_name, bins in bins_dict.items():
            if isinstance(bins, dict):
                edges = bins['edges']
                centers = bins['centers']
                attrs = bins.get('attrs', dict())
            else:
                edges = np.asarray(bins)
                centers = get_centers_from_edges(edges)
                attrs = dict()
            crd[crd_name] = dict(centers=centers, edges=edges, attrs=attrs)
        return crd

    def make(self, chunk):
        coord_names = list(self.coords.keys())
        bins = [self.coords[k]['edges'] for k in coord_names]
        coords = [chunk[k].values for k in coord_names]

        if 'weights' in chunk.variables:
            weights = chunk.weights.values
        else:
            weights = None

        values, idx = adaptive_histogram(coords, bins, weights=weights)
        yield dict(indices=idx, values=values)


def get_edges(a):
    half_offset = 0.5 * (a[1:] - a[:-1])
    first_edge = a[0] - half_offset[0]
    last_edge = a[-1] + half_offset[-1]
    mid_edges = a[:-1] + half_offset
    return np.concatenate([[first_edge], mid_edges, [last_edge]])


def get_centers_from_resolution_and_limits(resolution, limits):
    start, stop = limits

    # Check if limits is a datestring
    if isinstance(start, str) and isinstance(stop, str):
        try:
            start, stop = np.array([start, stop]).astype('datetime64')
        except ValueError:
            pass

    # Check if resolution is a timedelta specified as [value, unit]
    if np.issubdtype(np.array(start).dtype, np.datetime64):
        try:
            t64val, t64unit = resolution
            resolution = np.timedelta64(t64val, t64unit)
        except TypeError:
            pass

    centers = np.arange(start, stop + resolution, resolution)
    if centers[-1] > stop:
        centers = centers[:-1]
    return centers


def get_centers_from_edges(edges):
    edgediff = edges[1:] - edges[:-1]
    return edges[:-1] + 0.5 * edgediff


def adaptive_histogram(sample, bins, exact_dims=(), **kwargs):
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
    :param exact_dims:
    :return:
    """

    # Abort if there are no points in the sample
    if len(sample[0]) == 0:
        return np.zeros((0, ) * len(sample)), (slice(1, 0), ) * len(sample)

    # Cast datetime samples to be comparable with bins
    for i, s in enumerate(sample):
        s_dtype = np.asarray(s).dtype
        if np.issubdtype(s_dtype, np.datetime64):
            sample[i] = s.astype(bins[i].dtype)

    num_entries = next(len(s) for s in sample)
    included = np.ones(num_entries, dtype=bool)
    is_exact = np.zeros(len(bins), dtype=bool)
    is_exact[list(exact_dims)] = True

    # Find histogram coordinates of each entry
    binned_sample = []
    for s, b, ex in zip(sample, bins, is_exact):
        if ex:
            mapping = {k: i for i, k in enumerate(b)}
            coords = np.asarray([mapping.get(k, 0) for k in s])
            included = included & np.asarray([k in mapping for k in s])
        else:
            coords = np.searchsorted(b, s, side='right') - 1
            included = included & (0 <= coords) & (coords < len(b) - 1)
        binned_sample.append(coords)

    # Filter out coordinates outside interval
    for i, bs in enumerate(binned_sample):
        binned_sample[i] = bs[included]

    # Find min and max bin edges to be used
    idx = [(np.min(bs), np.max(bs) + 1) for bs in binned_sample]
    idx_slice = [slice(start, stop) for start, stop in idx]

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

    # Densify
    shifted_coords = coords - np.asarray([start for start, _ in idx])[:, np.newaxis]
    shape = [stop - start for start, stop in idx]
    hist_chunk = np.zeros(shape, dtype=vals.dtype)
    hist_chunk[tuple(shifted_coords)] = vals

    return hist_chunk, tuple(idx_slice)


def autobins(spec, dset):
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
    scan_output = {k: None for k in spec}
    if scan_params:
        scan_output = {**scan_output, **dset.scan(scan_params)}

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
