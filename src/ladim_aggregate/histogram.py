import numpy as np
import pandas as pd


class Histogrammer:
    def __init__(self, bins=None):
        self.bins = bins
        self.weights = dict(bincount=None)
        self.coords = Histogrammer._get_coords_from_bins(bins)

    @staticmethod
    def _get_coords_from_bins(bins_dict):
        crd = dict()
        for crd_name, bins in bins_dict.items():
            edges = np.asarray(bins)
            centers = get_centers_from_edges(edges)
            crd[crd_name] = dict(centers=centers, edges=edges)
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

    # Find histogram coordinates of each entry
    binned_sample = []
    for s, b in zip(sample, bins):
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

    return hist_chunk, idx_slice
