import numpy as np
from ladim_aggregate import histogram
import xarray as xr


class Test_Histogrammer:
    def test_can_generate_histogram_piece_from_chunk(self):
        h = histogram.Histogrammer(
            bins=dict(
                z=dict(edges=[-1.5, 1.5, 4.5], centers=[0, 3]),
                y=dict(edges=[-1, 1, 3, 5], centers=[0, 2, 4]),
                x=dict(edges=[-.5, .5, 1.5, 2.5, 3.5], centers=[0, 1, 2, 3]),
            )
        )
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
        h = histogram.Histogrammer(bins=dict(x=dict(
            edges=[0, 2, 6], centers=[1, 4],
        )))
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
        assert idx == np.s_[0:2, 0:3]
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
        assert idx == np.s_[0:2, 0:3]
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

    def test_returns_same_as_histogramdd_if_particles_outside_range(self):
        sample = [[1, 2, 3, 3, 4], [5, 6, 7, 8, 9]]
        bins = [[1.5, 2.5, 3.5], [6.5, 7.5, 8.5, 9.5]]
        hist_np, _ = np.histogramdd(sample, bins)

        hist2 = np.zeros([len(b) - 1 for b in bins])
        hist_chunk, idx = histogram.adaptive_histogram(sample, bins)
        assert idx == np.s_[1:2, 0:2]
        hist2[idx] = hist_chunk

        assert hist2.tolist() == hist_np.tolist()

    def test_returns_same_as_histogramdd_if_no_particles_in_range(self):
        sample = [[1, 2, 3, 3, 4], [5, 6, 7, 8, 9]]
        bins = [[10, 20], [30, 40, 50]]
        hist_np, _ = np.histogramdd(sample, bins)

        hist2 = np.zeros([len(b) - 1 for b in bins])
        hist_chunk, idx = histogram.adaptive_histogram(sample, bins)
        assert idx == np.s_[1:0, 1:0]
        hist2[idx] = hist_chunk

        assert hist2.tolist() == hist_np.tolist()


class Test_autobins:
    def test_computes_centers_if_spec_is_list(self):
        spec = dict(x=[1, 2, 3])
        bins = histogram.autobins(spec, dset=None)
        assert bins['x']['edges'].tolist() == [1, 2, 3]
        assert bins['x']['centers'].tolist() == [1.5, 2.5]

    def test_returns_verbatim_if_spec_is_edges_labels(self):
        spec = dict(x=dict(edges=[1, 2, 3], labels=[10, 20]))
        bins = histogram.autobins(spec, dset=None)
        assert bins['x']['edges'].tolist() == [1, 2, 3]
        assert bins['x']['centers'].tolist() == [10, 20]

    def test_returns_inclusive_range_if_spec_is_min_max_step(self):
        spec = dict(x=dict(min=1, max=10, step=3))
        bins = histogram.autobins(spec, dset=None)
        assert bins['x']['edges'].tolist() == [1, 4, 7, 10]
        assert bins['x']['centers'].tolist() == [2.5, 5.5, 8.5]

    def test_accepts_multiple_specs(self):
        spec_1 = dict(x=dict(min=1, max=10, step=3))
        bins_1 = histogram.autobins(spec_1, dset=None)
        spec_2 = dict(y=[1, 2, 3])
        bins_2 = histogram.autobins(spec_2, dset=None)
        spec = {**spec_1, **spec_2}
        bins = histogram.autobins(spec, dset=None)
        assert bins['x']['edges'].tolist() == bins_1['x']['edges'].tolist()
        assert bins['y']['edges'].tolist() == bins_2['y']['edges'].tolist()

    def test_returns_aligned_range_if_resolution(self):
        class MockLadimDataset:
            @staticmethod
            def scan(arg):
                assert arg == dict(x=['min', 'max'])
                return dict(x=dict(min=10, max=19))

        spec = dict(x=3)
        bins = histogram.autobins(spec, dset=MockLadimDataset())
        assert bins['x']['edges'].tolist() == [9, 12, 15, 18, 21]

    def test_returns_bins_if_unique(self):
        class MockLadimDataset:
            @staticmethod
            def scan(arg):
                assert arg == dict(x=['unique'])
                return dict(x=dict(unique=[1, 2, 5]))

        spec = dict(x='unique')
        bins = histogram.autobins(spec, dset=MockLadimDataset())
        assert bins['x']['edges'].tolist() == [1, 2, 5, 6]
        assert bins['x']['centers'].tolist() == [1, 2, 5]

    def test_copies_attributes_from_input_dataset(self):
        class MockLadimDataset:
            def __init__(self):
                self.attributes = dict(
                    x=dict(long_name="x coordinate value"),
                    y=dict(long_name="y coordinate value"),  # An extra attribute which is not in the bins
                )

        spec = dict(x=[1, 2, 3])
        bins = histogram.autobins(spec, dset=MockLadimDataset())
        assert bins['x']['attrs']['long_name'] == "x coordinate value"
        assert 'y' not in bins


class Test_align_to_resolution:
    def test_aligns_to_integer(self):
        assert histogram.align_to_resolution(value=22, resolution=3) == 21
        assert histogram.align_to_resolution(value=21, resolution=3) == 21
        assert histogram.align_to_resolution(value=19, resolution=3) == 18

    def test_aligns_to_time(self):
        align = histogram.align_to_resolution

        date = np.datetime64('2020-01-02T03:04:05')

        second = np.timedelta64(1, 's')
        minute = np.timedelta64(1, 'm')
        hour = np.timedelta64(1, 'h')
        day = np.timedelta64(1, 'D')

        assert align(date, second).astype(str) == '2020-01-02T03:04:05.000000'
        assert align(date, minute).astype(str) == '2020-01-02T03:04:00.000000'
        assert align(date, hour).astype(str) == '2020-01-02T03:00:00.000000'
        assert align(date, day).astype(str) == '2020-01-02T00:00:00.000000'


class Object:
    pass
