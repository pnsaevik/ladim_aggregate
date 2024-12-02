import pandas as pd

from ladim_aggregate import input as ladim_input
import numpy as np
import pytest
import xarray as xr
from unittest.mock import patch


@pytest.fixture(scope='module')
def ladim_dset():
    return xr.Dataset(
        data_vars=dict(
            X=xr.Variable('particle_instance', [5, 5, 6, 6, 5, 6]),
            Y=xr.Variable('particle_instance', [60, 60, 60, 61, 60, 62]),
            Z=xr.Variable('particle_instance', [0, 1, 2, 3, 4, 5],
                          attrs=dict(standard_name='depth')),
            lon=xr.Variable('particle_instance', [5, 5, 6, 6, 5, 6]),
            lat=xr.Variable('particle_instance', [60, 60, 60, 61, 60, 62]),
            instance_offset=xr.Variable((), 0),
            farm_id=xr.Variable('particle', [12345, 12346, 12347, 12348]),
            pid=xr.Variable('particle_instance', [0, 1, 2, 3, 1, 2]),
            particle_count=xr.Variable('time', [4, 2]),
        ),
        coords=dict(
            time=np.array(['2000-01-02', '2000-01-03']).astype('datetime64[ns]'),
        ),
    )


@pytest.fixture(scope='class')
def ladim_dset2(ladim_dset):
    d = ladim_dset.copy(deep=True)
    d['instance_offset'] += d.sizes['particle_instance']
    d = d.assign_coords(time=d.time + np.timedelta64(2, 'D'))
    return d


class Test_ladim_iterator:
    def test_returns_one_dataset_per_timestep_when_multiple_datasets(self, ladim_dset, ladim_dset2):
        it = ladim_input.ladim_iterator([ladim_dset, ladim_dset2])
        dsets = list(it)
        assert len(dsets) == ladim_dset.sizes['time'] + ladim_dset2.sizes['time']

    def test_returns_correct_time_selection(self, ladim_dset):
        iterator = ladim_input.ladim_iterator([ladim_dset])
        particle_count = [d.particle_count.values.tolist() for d in iterator]
        assert particle_count == [[4, 4, 4, 4], [2, 2]]

        iterator = ladim_input.ladim_iterator([ladim_dset])
        time = [d.time.values for d in iterator]
        assert len(time) == 2
        assert time[0].astype(str).tolist() == ['2000-01-02T00:00:00.000000000'] * 4
        assert time[1].astype(str).tolist() == ['2000-01-03T00:00:00.000000000'] * 2

    def test_returns_correct_instance_selection(self, ladim_dset):
        iterator = ladim_input.ladim_iterator([ladim_dset])
        z = [d.Z.values.tolist() for d in iterator]
        assert z == [[0, 1, 2, 3], [4, 5]]

        iterator = ladim_input.ladim_iterator([ladim_dset])
        pid = [d.pid.values.tolist() for d in iterator]
        assert pid == [[0, 1, 2, 3], [1, 2]]

    def test_broadcasts_particle_variables(self, ladim_dset):
        iterator = ladim_input.ladim_iterator([ladim_dset])
        farm_id = [d.farm_id.values.tolist() for d in iterator]
        assert farm_id == [[12345, 12346, 12347, 12348], [12346, 12347]]

    def test_updates_instance_offset(self, ladim_dset):
        iterator = ladim_input.ladim_iterator([ladim_dset])
        offset = [d.instance_offset.values.tolist() for d in iterator]
        assert offset == [0, 4]

    def test_accepts_timestep_selection_when_multiple_datasets(self, ladim_dset, ladim_dset2):
        # No timestep selection
        chunks = list(ladim_input.ladim_iterator([ladim_dset, ladim_dset2]))
        assert len(chunks) == 4

        times = [c.time.values[0].astype('datetime64[D]').astype(str) for c in chunks]
        assert times == ['2000-01-02', '2000-01-03', '2000-01-04', '2000-01-05']

        # Select first and fourth time step
        chunks = list(ladim_input.ladim_iterator([ladim_dset, ladim_dset2], timesteps=[0, 3]))
        assert len(chunks) == 2

        times = [c.time.values[0].astype('datetime64[D]').astype(str) for c in chunks]
        assert times == ['2000-01-02', '2000-01-05']

    def test_disregards_nonunique_timestep_selectors(self, ladim_dset):
        # Select first and second timestep
        chunks_1 = list(ladim_input.ladim_iterator([ladim_dset], timesteps=[0, 1]))
        assert len(chunks_1) == 2

        # Select multiple timesteps: We still get 2 chunks
        chunks_2 = list(ladim_input.ladim_iterator([ladim_dset], timesteps=[1, 0, 1]))
        assert len(chunks_2) == 2

        # And the time values are ordered
        assert chunks_2[0].time.values[0] < chunks_2[1].time.values[0]


class Test_LadimInputStream_scan:
    def test_can_return_min_value(self, ladim_dset):
        dset = ladim_input.LadimInputStream(ladim_dset)
        spec = dict(X=['min'])
        out = dset.scan(spec)
        assert out == dict(X=dict(min=5))

    def test_can_return_max_value(self, ladim_dset):
        dset = ladim_input.LadimInputStream(ladim_dset)
        spec = dict(X=['max'])
        out = dset.scan(spec)
        assert out == dict(X=dict(max=6))

    def test_can_return_multiple_stats(self, ladim_dset, ladim_dset2):
        ladim_dset3 = ladim_dset2.copy(deep=True)
        ladim_dset3['X'] += 10
        ladim_dset3['Y'] += 10
        dset = ladim_input.LadimInputStream([ladim_dset, ladim_dset3])
        spec = dict(X=['max'], Y=['min', 'max'])
        out = dset.scan(spec)
        assert out == dict(X=dict(max=16), Y=dict(min=60, max=72))


class Test_LadimInputStream_assign:
    def test_can_add_numexpr_variables(self, ladim_dset):
        dset = ladim_input.LadimInputStream(ladim_dset)
        dset.add_derived_variable('sumcoords', "X + Y")
        chunk = next(dset.chunks())
        assert chunk.sumcoords.values.tolist() == (chunk.X + chunk.Y).values.tolist()

    def test_can_add_agg_variable(self, ladim_dset):
        dset = ladim_input.LadimInputStream(ladim_dset)
        dset.add_aggregation_variable('X', 'max')
        dset.add_aggregation_variable('Y', 'min')
        chunk = next(dset.chunks())
        assert chunk.MAX_X.values.tolist() == ladim_dset.X.max().values.tolist()
        assert chunk.MIN_Y.values.tolist() == ladim_dset.Y.min().values.tolist()

    def test_can_add_unique(self, ladim_dset):
        dset = ladim_input.LadimInputStream(ladim_dset)
        dset.add_aggregation_variable('X', 'unique')
        unique_x = dset.get_aggregation_value('UNIQUE_X')
        assert unique_x.values.tolist() == np.unique(ladim_dset.X.values).tolist()


class Test_LadimInputStream:
    def test_can_initialise_from_xr_dataset(self, ladim_dset):
        dset = ladim_input.LadimInputStream(ladim_dset)
        next(dset.chunks())

    def test_can_initialise_from_multiple_xr_datasets(self, ladim_dset, ladim_dset2):
        dset = ladim_input.LadimInputStream([ladim_dset, ladim_dset2])
        next(dset.chunks())

    def test_reads_one_timestep_at_the_time(self, ladim_dset, ladim_dset2):
        dset = ladim_input.LadimInputStream([ladim_dset, ladim_dset2])
        pids = list(c.pid.values.tolist() for c in dset.chunks())
        assert len(pids) == ladim_dset.sizes['time'] + ladim_dset2.sizes['time']
        assert pids == [[0, 1, 2, 3], [1, 2], [0, 1, 2, 3], [1, 2]]

    def test_broadcasts_time_vars_when_reading(self, ladim_dset, ladim_dset2):
        dset = ladim_input.LadimInputStream([ladim_dset, ladim_dset2])
        counts = list(c.particle_count.values.tolist() for c in dset.chunks())
        assert counts == [[4, 4, 4, 4], [2, 2], [4, 4, 4, 4], [2, 2]]

    def test_broadcasts_particle_vars_when_reading(self, ladim_dset, ladim_dset2):
        dset = ladim_input.LadimInputStream([ladim_dset, ladim_dset2])
        farmid = list(c.farm_id.values.tolist() for c in dset.chunks())
        assert farmid == [
            [12345, 12346, 12347, 12348],
            [12346, 12347],
            [12345, 12346, 12347, 12348],
            [12346, 12347],
        ]

    def test_can_apply_filter_string(self, ladim_dset):
        dset = ladim_input.LadimInputStream(ladim_dset)

        # No filter: 6 particle instances
        chunks = xr.concat(dset.chunks(), dim='pid')
        assert chunks.sizes['pid'] == 6

        # With filter: 4 particle instances
        filters = "farm_id != 12346"
        chunks = xr.concat(dset.chunks(filters=filters), dim='pid')
        assert chunks.sizes['pid'] == 4

        # A more complex filter expression
        filters = "(farm_id > 12345) & (farm_id < 12347)"
        chunks = xr.concat(dset.chunks(filters=filters), dim='pid')
        assert chunks.sizes['pid'] == 2

    def test_can_apply_particle_filter(self, ladim_dset):
        dset = ladim_input.LadimInputStream(ladim_dset)

        # No filter: 6 particle instances
        chunks = xr.concat(dset.chunks(), dim='pid')
        assert chunks.sizes['pid'] == 6

        # With filter: Keeps only the first time the condition is triggered
        filters = "Z >= 2"
        chunks = xr.concat(dset.chunks(particle_filter=filters), dim='pid')
        assert chunks['pid'].values.tolist() == [2, 3, 1]
        assert chunks['Z'].values.tolist() == [2, 3, 4]

        # Combination of particle filter and regular filter
        chunk_iter = dset.chunks(particle_filter="Z >= 2", filters="pid != 3")
        chunks = xr.concat(chunk_iter, dim='pid')
        assert chunks['pid'].values.tolist() == [2, 1]
        assert chunks['Z'].values.tolist() == [2, 4]

    def test_can_apply_timestep_filter(self, ladim_dset):
        dset = ladim_input.LadimInputStream(ladim_dset)

        # No filter: 6 particle instances
        chunks = xr.concat(dset.chunks(), dim='pid')
        assert chunks.sizes['pid'] == 6

        # Just first timestep: 4 particle instances
        chunks = xr.concat(dset.chunks(timestep_filter=[0]), dim='pid')
        assert chunks.sizes['pid'] == 4

        # Just second timestep: 2 particle instances
        chunks = xr.concat(dset.chunks(timestep_filter=[1]), dim='pid')
        assert chunks.sizes['pid'] == 2

    def test_can_add_weights_from_string_expression(self, ladim_dset):
        dset = ladim_input.LadimInputStream(ladim_dset)
        dset.add_derived_variable('weights', 'X + Y')
        chunk = next(c for c in dset.chunks())
        assert 'weights' in chunk
        assert len(chunk['weights']) > 0
        assert chunk['weights'].values.tolist() == list(
            chunk['X'].values + chunk['Y'].values
        )

    def test_can_return_attributes_of_variables(self, ladim_dset):
        dset = ladim_input.LadimInputStream(ladim_dset)
        assert dset.attributes['Z']['standard_name'] == 'depth'


class Test_LadimInputStream_grid:
    def test_can_add_grid_variables(self, ladim_dset):
        dset = ladim_input.LadimInputStream(ladim_dset)
        grid = xr.DataArray(
            data=[100, 200, 300],
            coords=dict(Z=[0, 4, 8]),
            name="multiplier",
        )
        dset.add_grid_variable(grid, 'linear')
        chunk = next(dset.chunks())
        assert "multiplier" in chunk.variables
        assert chunk.multiplier.values.tolist() == [100, 125, 150, 175]

    def test_can_use_interp_method_nearest(self, ladim_dset):
        dset = ladim_input.LadimInputStream(ladim_dset)
        grid = xr.DataArray(
            data=[100, 200, 300],
            coords=dict(Z=[0, 4, 8]),
            name="multiplier",
        )
        dset.add_grid_variable(grid, 'nearest')
        chunk = next(dset.chunks())
        assert "multiplier" in chunk.variables
        assert chunk.multiplier.values.tolist() == [100, 100, 100, 200]

    def test_can_use_interp_method_bin(self, ladim_dset):
        dset = ladim_input.LadimInputStream(ladim_dset)
        grid = xr.DataArray(
            data=[100, 200, 300],
            coords=dict(Z=[0, 4, 8]),
            name="multiplier",
        )
        dset.add_grid_variable(grid, 'bin')
        chunk = next(dset.chunks())
        assert "multiplier" in chunk.variables
        assert chunk.multiplier.values.tolist() == [100, 100, 100, 100]


class Test_update_agg:
    def test_can_compute_max(self):
        assert ladim_input.update_agg(None, 'max', [1, 2, 3]) == 3
        assert ladim_input.update_agg(4, 'max', [1, 2, 3]) == 4
        assert ladim_input.update_agg(2, 'max', [1, 2, 3]) == 3

    def test_can_compute_min(self):
        assert ladim_input.update_agg(None, 'min', [1, 2, 3]) == 1
        assert ladim_input.update_agg(0, 'min', [1, 2, 3]) == 0
        assert ladim_input.update_agg(2, 'min', [1, 2, 3]) == 1

    def test_can_compute_unique(self):
        assert ladim_input.update_agg(None, 'unique', [1, 1, 3]) == [1, 3]
        assert ladim_input.update_agg([], 'unique', [1, 1, 3]) == [1, 3]
        assert ladim_input.update_agg([2, 3, 4], 'unique', [1, 1, 3]) == [1, 2, 3, 4]

    def test_can_update_init_if_old_data_is_supplied(self):
        old_data = np.array([1, 0, 0, 0, 0])
        old_mask = (old_data > 0)
        new_data = np.array([5, 6, 7, 8])
        new_pid = np.array([0, 2, 0, 2])

        data, mask = ladim_input.update_agg(
            old=(old_data, old_mask), aggfun='init', data=(new_data, new_pid))

        assert data.tolist() == [1, 0, 6, 0, 0]
        assert mask.tolist() == (data > 0).tolist()

    def test_can_update_final_if_old_data_is_supplied(self):
        old_data = np.array([1, 0, 0, 0, 0])
        old_mask = (old_data > 0)
        new_data = np.array([3, 4, 5, 6])
        new_pid = np.array([0, 2, 0, 2])

        data, mask = ladim_input.update_agg(
            old=(old_data, old_mask), aggfun='final', data=(new_data, new_pid))

        assert data.tolist() == [5, 0, 6, 0, 0]
        assert mask.tolist() == (data > 0).tolist()

    def test_can_initialize_init(self):
        new_data = np.array([5, 6, 7, 8])
        new_pid = np.array([0, 2, 0, 2])

        data, mask = ladim_input.update_agg(
            old=None, aggfun='init', data=(new_data, new_pid))

        assert data.tolist() == [5, 0, 6]
        assert mask.tolist() == (data > 0).tolist()

    def test_can_initialize_final(self):
        new_data = np.array([5, 6, 7, 8])
        new_pid = np.array([0, 2, 0, 2])

        data, mask = ladim_input.update_agg(
            old=None, aggfun='final', data=(new_data, new_pid))

        assert data.tolist() == [7, 0, 8]
        assert mask.tolist() == (data > 0).tolist()


class Test_create_pfilter:
    def test_checks_if_variable_is_triggered(self):
        pfilter = ladim_input.create_pfilter(spec='age >= 2')

        chunk = xr.Dataset(
            data_vars=dict(
                age=xr.Variable('particle_instance', [0, 1, 2, 3, 4, 5]),
                pid=xr.Variable('particle_instance', [6, 7, 8, 9, 10, 12])
            ),
        )
        is_triggered = pfilter(chunk)

        # First two particles are not triggered, the remaining are
        assert is_triggered.values.tolist() == [0, 0, 1, 1, 1, 1]

    def test_counts_only_first_occurrence(self):
        pfilter = ladim_input.create_pfilter(spec='age >= 2')

        chunk = xr.Dataset(
            data_vars=dict(
                pid=xr.Variable('particle_instance', [0, 1, 2, 3, 4, 5])
            ),
        )
        chunk1 = chunk.assign(age=xr.Variable('particle_instance', [0, 0, 0, 0, 4, 5]))
        chunk2 = chunk.assign(age=xr.Variable('particle_instance', [2, 0, 0, 0, 6, 6]))
        chunk3 = chunk.assign(age=xr.Variable('particle_instance', [0, 2, 0, 0, 9, 9]))

        pfilter(chunk1)
        pfilter(chunk2)
        is_triggered = pfilter(chunk3)
        assert is_triggered.values.tolist() == [0, 1, 0, 0, 0, 0]

    def test_works_with_large_pids(self):
        pfilter = ladim_input.create_pfilter(spec='age >= 2')

        chunk = xr.Dataset(
            data_vars=dict(
                age=xr.Variable('particle_instance', [0, 1, 2]),
                pid=xr.Variable('particle_instance', [1001, 1002, 1003])
            ),
        )

        pfilter(chunk)


class Test_create_varfunc:
    @patch('ladim_aggregate.input.get_varfunc_from_funcstring')
    def test_calls_correct_function_when_singledot_spec(self, mock_class):
        _ = ladim_input.create_varfunc('numpy.sum')
        assert mock_class.call_count == 1

    @patch('ladim_aggregate.input.get_varfunc_from_callable')
    def test_calls_correct_function_when_callable(self, mock_class):
        _ = ladim_input.create_varfunc(np.sum)
        assert mock_class.call_count == 1

    @patch('ladim_aggregate.input.get_varfunc_from_funcstring')
    def test_calls_correct_function_when_doubledot_spec(self, mock_class):
        _ = ladim_input.create_varfunc('numpy.linalg.inv')
        assert mock_class.call_count == 1

    @patch('ladim_aggregate.input.get_varfunc_from_numexpr')
    def test_calls_correct_function_when_numexpr_with_ints(self, mock_class):
        _ = ladim_input.create_varfunc('myvar * 123')
        assert mock_class.call_count == 1

    @patch('ladim_aggregate.input.get_varfunc_from_numexpr')
    def test_calls_correct_function_when_varname(self, mock_class):
        _ = ladim_input.create_varfunc('my_var')
        assert mock_class.call_count == 1

    @patch('ladim_aggregate.input.get_varfunc_from_numexpr')
    def test_calls_correct_function_when_pure_number(self, mock_class):
        _ = ladim_input.create_varfunc('1')
        assert mock_class.call_count == 1
        _ = ladim_input.create_varfunc(1)
        assert mock_class.call_count == 2
        _ = ladim_input.create_varfunc(1.0)
        assert mock_class.call_count == 3

    @patch('ladim_aggregate.input.get_varfunc_from_numexpr')
    def test_calls_correct_function_when_numexpr_with_floats(self, mock_class):
        _ = ladim_input.create_varfunc('myvar * 1.23')
        assert mock_class.call_count == 1


class Test_get_varfunc_from_callable:
    def test_returns_weight_function(self):
        weight_fn = ladim_input.get_varfunc_from_callable(lambda a, b: a + b)
        chunk = xr.Dataset(dict(a=[1, 2, 3], b=[4, 5, 6]))
        new_var = weight_fn(chunk)
        assert new_var.values.tolist() == [5, 7, 9]

    def test_fails_if_param_not_in_chunk(self):
        weight_fn = ladim_input.get_varfunc_from_callable(lambda a, b: a + b)
        chunk = xr.Dataset(dict(a=[1, 2, 3]))
        with pytest.raises(KeyError):
            _ = weight_fn(chunk)

    def test_can_use_optional_params(self):
        def fn(a, b, c=1):
            return a + b + c

        weight_fn = ladim_input.get_varfunc_from_callable(fn)
        chunk_1 = xr.Dataset(dict(a=[1, 2, 3], b=[4, 5, 6]))
        chunk_2 = xr.Dataset(dict(a=[1, 2, 3], b=[4, 5, 6], c=[2, 2, 2]))

        new_var_1 = weight_fn(chunk_1)
        assert new_var_1.values.tolist() == [6, 8, 10]

        new_var_2 = weight_fn(chunk_2)
        assert new_var_2.values.tolist() == [7, 9, 11]
