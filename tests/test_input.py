from ladim_aggregate import input as ladim_input
import numpy as np
import pytest
import xarray as xr


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
            time=np.array(['2000-01-02', '2000-01-03']).astype('datetime64[D]'),
        ),
    )


@pytest.fixture(scope='class')
def ladim_dset2(ladim_dset):
    d = ladim_dset.copy(deep=True)
    d['instance_offset'] += d.dims['particle_instance']
    d = d.assign_coords(time=d.time + np.timedelta64(2, 'D'))
    return d


class Test_ladim_iterator:
    def test_returns_one_dataset_per_timestep_when_multiple_datasets(self, ladim_dset, ladim_dset2):
        it = ladim_input.ladim_iterator([ladim_dset, ladim_dset2])
        dsets = list(it)
        assert len(dsets) == ladim_dset.dims['time'] + ladim_dset2.dims['time']

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
        assert len(pids) == ladim_dset.dims['time'] + ladim_dset2.dims['time']
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
        assert chunks.dims['pid'] == 6

        # With filter: 4 particle instances
        filters = "farm_id != 12346"
        chunks = xr.concat(dset.chunks(filters=filters), dim='pid')
        assert chunks.dims['pid'] == 4

        # A more complex filter expression
        filters = "(farm_id > 12345) & (farm_id < 12347)"
        chunks = xr.concat(dset.chunks(filters=filters), dim='pid')
        assert chunks.dims['pid'] == 2

    def test_can_add_weights_from_string_expression(self, ladim_dset):
        dset = ladim_input.LadimInputStream(ladim_dset)
        chunk = next(c for c in dset.chunks(newvars=dict(weights='X + Y')))
        assert 'weights' in chunk
        assert len(chunk['weights']) > 0
        assert chunk['weights'].values.tolist() == list(
            chunk['X'].values + chunk['Y'].values
        )

    def test_can_return_attributes_of_variables(self, ladim_dset):
        dset = ladim_input.LadimInputStream(ladim_dset)
        assert dset.attributes['Z']['standard_name'] == 'depth'


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
