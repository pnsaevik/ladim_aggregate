import numpy as np


class Histogrammer:
    def __init__(self, resolution, limits):
        self.resolution = resolution
        self.limits = limits
        self.weights = dict(bincount=None)
        self.coords = Histogrammer._get_coords(resolution, limits)

    @staticmethod
    def get_edges(centers):
        return get_edges(centers)

    @staticmethod
    def get_centers(resolution, limits):
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

    @staticmethod
    def _get_coords(resolution_dict, limits_dict):
        crd = dict()
        for crd_name, resolution in resolution_dict.items():
            limits = limits_dict[crd_name]
            centers = Histogrammer.get_centers(resolution, limits)
            edges = Histogrammer.get_edges(centers)
            crd[crd_name] = dict(centers=centers, edges=edges)
        return crd

    @staticmethod
    def adaptive_histogram(sample, bins, **kwargs):
        """
        Return an adaptive histogram

        For input values `sample` and `bins`, the code snippet

        hist = np.zeros([len(b) for b in bins])
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
            if np.issubdtype(s.dtype, np.datetime64):
                sample[i] = s.astype(bins[i].dtype)

        # Find min and max bin edges to be used
        idx = []
        bins_subset = []
        for coord, bin_edges in zip(sample, bins):
            digitized_min = np.searchsorted(bin_edges, np.min(coord), side='right')
            digitized_max = np.searchsorted(bin_edges, np.max(coord), side='right')
            idx_start = max(0, digitized_min - 1)
            idx_stop = min(len(bin_edges), digitized_max + 1)
            idx.append(slice(idx_start, idx_stop - 1))
            bins_subset.append(bin_edges[idx_start:idx_stop])

        rasterized_data = np.histogramdd(sample, bins_subset, **kwargs)[0]
        return rasterized_data, tuple(idx)

    def make(self, chunk):
        coord_names = list(self.coords.keys())
        bins = [self.coords[k]['edges'] for k in coord_names]
        coords = [chunk[k].values for k in coord_names]

        if 'weights' in chunk.variables:
            weights = chunk.weights.values
        else:
            weights = None

        values, idx = self.adaptive_histogram(coords, bins, weights=weights)
        yield dict(indices=idx, values=values)


def get_edges(a):
    half_offset = 0.5 * (a[1:] - a[:-1])
    first_edge = a[0] - half_offset[0]
    last_edge = a[-1] + half_offset[-1]
    mid_edges = a[:-1] + half_offset
    return np.concatenate([[first_edge], mid_edges, [last_edge]])
