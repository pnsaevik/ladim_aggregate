import numpy as np
from ladim_aggregate import histogram
import xarray as xr


class Test_Histogrammer:
    def test_can_generate_histogram_piece_from_chunk(self):
        h = histogram.Histogrammer(
            bins=dict(z=[-1.5, 1.5, 4.5], y=[-1, 1, 3, 5], x=[-.5, .5, 1.5, 2.5, 3.5]))
        chunk = xr.Dataset(dict(x=[0, 1, 3], y=[0, 2, 4], z=[0, 1, 3]))
        hist_piece = next(h.make(chunk))
        assert hist_piece['values'].tolist() == [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
        ]
        start = [idx.start for idx in hist_piece['indices']]
        assert start == [0, 0, 0]
        stop = [idx.stop for idx in hist_piece['indices']]
        assert stop == [2, 3, 4]

    def test_can_generate_weighted_histogram_piece_from_chunk(self):
        h = histogram.Histogrammer(bins=dict(x=[0, 2, 6]))
        chunk = xr.Dataset(dict(x=[1, 3, 5], weights=[10, 100, 1000]))
        hist_piece = next(h.make(chunk))
        assert hist_piece['values'].tolist() == [10, 1100]


class Test_adaptive_histogram:
    def test_returns_same_as_histogramdd_if_count(self):
        sample = [
            [1.5, 2.5, 3.5, 4.5, 5.5],
            [6.5, 6.5, 7.5, 8.5, 9.5],
        ]
        bins = [[1, 4, 6], [6, 8, 9, 10, 12]]

        hist, _ = np.histogramdd(sample, bins)

        hist2 = np.zeros([len(b) - 1 for b in bins])
        hist_chunk, idx = histogram.adaptive_histogram(sample, bins)
        hist2[idx] = hist_chunk

        assert hist2.tolist() == hist.tolist()

    def test_returns_only_partial_matrix(self):
        sample = [
            [1.5, 2.5, 3.5, 4.5, 5.5],
            [6.5, 6.5, 7.5, 8.5, 9.5],
        ]
        bins = [[1, 4, 6, 8], [6, 8, 9, 10, 12]]

        hist_chunk, idx = histogram.adaptive_histogram(sample, bins)

        assert hist_chunk.shape[0] < len(bins[0]) - 1
        assert hist_chunk.shape[1] < len(bins[1]) - 1

    def test_returns_same_as_histogramdd_if_weights(self):
        sample = [
            [1.5, 2.5, 3.5, 4.5, 5.5],
            [6.5, 6.5, 7.5, 8.5, 9.5],
        ]
        weights = [1, 2, 3, 4, 5]
        bins = [[1, 4, 6], [6, 8, 9, 10, 12]]

        hist, _ = np.histogramdd(sample, bins, weights=weights)

        hist2 = np.zeros([len(b) - 1 for b in bins])
        hist_chunk, idx = histogram.adaptive_histogram(sample, bins, weights=weights)
        hist2[idx] = hist_chunk

        assert hist2.tolist() == hist.tolist()

    def test_returns_same_as_histogramdd_if_no_particles(self):
        sample = [[], []]
        bins = [[1, 4, 6], [6, 8, 9, 10, 12]]

        hist, _ = np.histogramdd(sample, bins)

        hist2 = np.zeros([len(b) - 1 for b in bins])
        hist_chunk, idx = histogram.adaptive_histogram(sample, bins)
        hist2[idx] = hist_chunk

        assert hist2.tolist() == hist.tolist()
