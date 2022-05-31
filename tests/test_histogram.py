from ladim_aggregate import histogram
import xarray as xr


class Test_Histogrammer:
    def test_can_compute_centers_from_resolution_and_limits(self):
        h = histogram.Histogrammer(
            resolution=dict(x=1, y=2, z=4),
            limits=dict(x=[0, 4], y=[1, 10], z=[-10, 10]),
        )
        assert h.coords['x']['centers'].tolist() == [0, 1, 2, 3, 4]
        assert h.coords['y']['centers'].tolist() == [1, 3, 5, 7, 9]
        assert h.coords['z']['centers'].tolist() == [-10, -6, -2, 2, 6, 10]

    def test_can_compute_edges_from_resolution_and_limits(self):
        h = histogram.Histogrammer(
            resolution=dict(x=1, y=2, z=4),
            limits=dict(x=[0, 4], y=[1, 10], z=[-10, 10]),
        )
        assert h.coords['x']['edges'].tolist() == [-.5, .5, 1.5, 2.5, 3.5, 4.5]
        assert h.coords['y']['edges'].tolist() == [0, 2, 4, 6, 8, 10]
        assert h.coords['z']['edges'].tolist() == [-12, -8, -4, 0, 4, 8, 12]

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
