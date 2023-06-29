import contextlib
import datetime
import numpy as np
import pytest
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
        chunk = xr.Dataset(dict(x=[1, 3, 5], _auto_weights=[10, 100, 1000]))
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
        # Weights are included since a previous bug caused the function to
        # fail when weighted particles were outside range
        sample = [[1, 2, 3, 3, 4], [5, 6, 7, 8, 9]]
        weights = [1, 1, 1, 1, 1]
        bins = [[1.5, 2.5, 3.5], [6.5, 7.5, 8.5, 9.5]]
        hist_np, _ = np.histogramdd(sample, bins, weights=weights)

        hist2 = np.zeros([len(b) - 1 for b in bins])
        hist_chunk, idx = histogram.adaptive_histogram(sample, bins, weights=weights)
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
            def __init__(self):
                self._specials = {'MIN_x': 10, 'MAX_x': 19}
                self.add_aggregation_variable = lambda v, op: f'{op.upper()}_{v}'

            def get_aggregation_value(self, key):
                return self._specials[key]

        spec = dict(x=3)
        bins = histogram.autobins(spec, dset=MockLadimDataset())
        assert bins['x']['edges'].tolist() == [9, 12, 15, 18, 21]

    def test_returns_bins_if_unique(self):
        class MockLadimDataset:
            def __init__(self):
                self._specials = {'UNIQUE_x': [1, 2, 5]}
                self.add_aggregation_variable = lambda v, op: f'{op.upper()}_{v}'

            def get_aggregation_value(self, key):
                return self._specials[key]

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

    def test_converts_time_specs(self):
        class MockLadimDataset:
            @contextlib.contextmanager
            def open_dataset(self, _):
                attrs = dict(units='hours since 1980-01-01')
                tvar = xr.Variable(dims='time', data=[], attrs=attrs)
                yield xr.Dataset(data_vars=dict(time=tvar))

        spec = dict(time=dict(min='1980-01-01', max='1980-01-03', step='1 days'))
        bins = histogram.autobins(spec, dset=MockLadimDataset())
        assert bins['time']['centers'].tolist() == [12, 36]
        assert bins['time']['edges'].tolist() == [0, 24, 48]


class Test_convert_datebins:
    @pytest.fixture(scope='class')
    def time_dset(self):
        attrs = dict(units='days since 2000-01-02')
        tvar = xr.Variable(dims='time', data=[1, 3, 5], attrs=attrs)
        xvar = xr.Variable(dims='time', data=[10, 30, 50])
        return xr.Dataset(data_vars=dict(time=tvar, X=xvar))

    def test_converts_only_datevars(self, time_dset):
        spec = dict(
            time=['2000-01', '2000-02', '2000-03'],
            X=[0, 10, 20, 30],
        )
        newspec = histogram.convert_datebins(spec, time_dset)
        assert newspec == dict(
            time=[-1, 30, 59],
            X=[0, 10, 20, 30],
        )

    def test_ignores_nonexisting_vars(self, time_dset):
        spec = dict(Y=[0, 10, 20, 30])
        newspec = histogram.convert_datebins(spec, time_dset)
        assert newspec == spec


class Test_convert_binspec:
    def test_converts_edgeformat(self):
        spec = ['1970-01', '1970-02', '1970-03', 100]
        result = histogram.convert_binspec(spec, 'days since 1970-01-01', 'standard')
        assert result == [0, 31, 31+28, 100]

    def test_converts_rangeformat(self):
        spec = dict(min='1970-01', max='1970-02', step='48 hours')
        result = histogram.convert_binspec(spec, 'days since 1970-01-01', 'standard')
        assert result == dict(min=0, max=31, step=2)

    def test_converts_labelformat(self):
        spec = dict(edges=['1970-01', '1970-02', '1970-03'], labels=['jan', 'feb'])
        result = histogram.convert_binspec(spec, 'days since 1970-01-01', 'standard')
        assert result == dict(edges=[0, 31, 31+28], labels=['jan', 'feb'])

    def test_converts_cfstring_format(self):
        spec = '48 hours'
        result = histogram.convert_binspec(spec, 'days since 2020-01-01', 'standard')
        assert result == 2


class Test_convert_step:
    def test_returns_integers_verbatim(self):
        result = histogram.convert_step(123, 'days since 1980-01-01', 'standard')
        assert result == 123

    def test_converts_string_spec(self):
        result = histogram.convert_step('48 hours', 'days since 1980-01-01', 'standard')
        assert result == 2

    def test_converts_string_spec_when_timedate(self):
        result = histogram.convert_step('1 days', 'hours since 1980-01-01 01:00:00', 'standard')
        assert result == 24


class Test_convert_date:
    def test_returns_integers_verbatim(self):
        result = histogram.convert_date(123, 'days since 1970-01-01', 'standard')
        assert result == 123

    def test_returns_floats_verbatim(self):
        result = histogram.convert_date(123.5, 'days since 1970-01-01', 'standard')
        assert result == 123.5

    def test_converts_dates(self):
        date = datetime.date(2000, 5, 3)
        result = histogram.convert_date(date, 'days since 1970-01-01', 'standard')
        assert result == 11080

    def test_converts_datetimes(self):
        date = datetime.datetime(2000, 5, 3, 12, 0, 0)
        result = histogram.convert_date(date, 'days since 1970-01-01', 'standard')
        assert result == 11080

    def test_converts_datetime64(self):
        date = np.datetime64('1970-02')
        result = histogram.convert_date(date, 'days since 1970-01-01', 'standard')
        assert result == 31

    def test_converts_string(self):
        date = '1970-02'
        result = histogram.convert_date(date, 'days since 1970-01-01', 'standard')
        assert result == 31


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


class Test_t64conv:
    def test_returns_timedeltas_verbatim(self):
        t = np.timedelta64(1, 's')
        result = histogram.t64conv(t)
        assert result is t

    def test_returns_numbers_verbatim(self):
        t = 123
        result = histogram.t64conv(t)
        assert result is t

    def test_converts_tuple_values(self):
        t = [23, 's']
        result = histogram.t64conv(t)
        assert result == np.timedelta64(23, 's')

    def test_converts_cftime_strings(self):
        t_vec = [
            '1 day',
            '24 hours',
            '24 h',
            '20 minutes',
            '23 seconds',
            '23 s',
            '23000 ms',
            '23000000 us',
        ]
        result = [histogram.t64conv(t) for t in t_vec]
        assert [r.astype(str) for r in result] == [
            '1 days',
            '24 hours',
            '24 hours',
            '20 minutes',
            '23 seconds',
            '23 seconds',
            '23000 milliseconds',
            '23000000 microseconds',
        ]


class Object:
    pass
