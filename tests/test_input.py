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
            Z=xr.Variable('particle_instance', [0, 1, 2, 3, 4, 5]),
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


@pytest.fixture(scope='module')
def fnames():
    import pkg_resources
    try:
        yield dict(
            outdata=pkg_resources.resource_filename('ladim_plugins.chemicals', 'out.nc'),
        )
    finally:
        pkg_resources.cleanup_resources()


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


class Test_LadimInputStream:
    def test_can_initialise_from_xr_dataset(self, ladim_dset):
        with ladim_input.LadimInputStream(ladim_dset) as dset:
            dset.read()

    def test_can_initialise_from_multiple_xr_datasets(self, ladim_dset, ladim_dset2):
        with ladim_input.LadimInputStream([ladim_dset, ladim_dset2]) as dset:
            dset.read()

    def test_can_initialise_from_filename(self, fnames):
        ladim_fname = fnames['outdata']
        with ladim_input.LadimInputStream(ladim_fname) as dset:
            dset.read()

    def test_can_initialise_from_multiple_filenames(self, fnames):
        ladim_fname = fnames['outdata']
        with ladim_input.LadimInputStream([ladim_fname, ladim_fname]) as dset:
            dset.read()

    def test_can_seek_to_dataset_beginning(self, ladim_dset):
        with ladim_input.LadimInputStream(ladim_dset) as dset:
            first_chunk = dset.read()
            second_chunk = dset.read()
            dset.seek(0)
            third_chunk = dset.read()

        assert first_chunk.pid.values.tolist() != second_chunk.pid.values.tolist()
        assert first_chunk.pid.values.tolist() == third_chunk.pid.values.tolist()

    def test_reads_one_timestep_at_the_time(self, ladim_dset, ladim_dset2):
        with ladim_input.LadimInputStream([ladim_dset, ladim_dset2]) as dset:
            pids = list(c.pid.values.tolist() for c in dset.chunks())
            assert len(pids) == ladim_dset.dims['time'] + ladim_dset2.dims['time']
            assert pids == [[0, 1, 2, 3], [1, 2], [0, 1, 2, 3], [1, 2]]

    def test_broadcasts_time_vars_when_reading(self, ladim_dset, ladim_dset2):
        with ladim_input.LadimInputStream([ladim_dset, ladim_dset2]) as dset:
            counts = list(c.particle_count.values.tolist() for c in dset.chunks())
            assert counts == [[4, 4, 4, 4], [2, 2], [4, 4, 4, 4], [2, 2]]

    def test_broadcasts_particle_vars_when_reading(self, ladim_dset, ladim_dset2):
        with ladim_input.LadimInputStream([ladim_dset, ladim_dset2]) as dset:
            farmid = list(c.farm_id.values.tolist() for c in dset.chunks())
            assert farmid == [
                [12345, 12346, 12347, 12348],
                [12346, 12347],
                [12345, 12346, 12347, 12348],
                [12346, 12347],
            ]

    def test_autolimit_aligns_to_wholenumber_resolution_points(self, ladim_dset, ladim_dset2):
        with ladim_input.LadimInputStream([ladim_dset, ladim_dset2]) as dset:
            assert dset.find_limits(dict(X=1)) == dict(X=[5, 7])
            assert dset.find_limits(dict(X=2)) == dict(X=[4, 8])
            assert dset.find_limits(dict(X=10)) == dict(X=[0, 10])

    def test_autolimit_aligns_to_wholenumber_resolution_points_when_time(self, ladim_dset, ladim_dset2):
        with ladim_input.LadimInputStream([ladim_dset, ladim_dset2]) as dset:
            limits = dset.find_limits(dict(time=np.timedelta64(1, 'h')))
            assert list(limits.keys()) == ['time']
            timestr = np.array(limits['time']).astype('datetime64[h]').astype(str)
            assert timestr.tolist() == ['2000-01-02T00', '2000-01-05T01']

            limits = dset.find_limits(dict(time=np.timedelta64(6, 'h')))
            assert list(limits.keys()) == ['time']
            timestr = np.array(limits['time']).astype('datetime64[h]').astype(str)
            assert timestr.tolist() == ['2000-01-02T00', '2000-01-05T06']
