import subprocess
from ladim_aggregate.script import SCRIPT_NAME
from ladim_aggregate import script, examples
import pytest
import netCDF4 as nc
import xarray as xr
from uuid import uuid4


class Test_main:
    def test_prints_help_message_when_no_arguments(self, capsys):
        script.main()
        out = capsys.readouterr().out
        assert out.startswith('usage: ' + SCRIPT_NAME)

    def test_prints_help_message_when_help_argument(self, capsys):
        with pytest.raises(SystemExit):
            script.main('--help')
        out = capsys.readouterr().out
        assert out.startswith('usage: ' + SCRIPT_NAME)


named_examples_all = examples.Example.available()
named_examples = ['grid_2D']


class Test_command_line_script:
    @pytest.mark.parametrize("name", named_examples)
    def test_can_extract_and_run_example(self, name, tmp_path):
        import os
        os.chdir(tmp_path)
        r = subprocess.run([SCRIPT_NAME, '--example', name], stdout=subprocess.PIPE)
        assert r.stdout.decode('utf-8') == ''
        files = {f.name for f in tmp_path.glob('*')}
        assert 'aggregate.yaml' in files
        assert 'count.nc' in files


class Test_run_conf:
    @pytest.fixture
    def infile(self):
        dset = xr.Dataset(
            coords=dict(
                time=[100, 200],
            ),
            data_vars=dict(
                particle_count=xr.Variable('time', [1, 2]),
                x=xr.Variable('particle_instance', [7, 13, 15]),
                pid=xr.Variable('particle_instance', [0, 0, 1]),
            ),
        )
        return dset

    @pytest.fixture
    def outfile(self):
        with nc.Dataset(uuid4(), 'w', diskless=True) as dset:
            yield dset

    @pytest.fixture
    def gridfile(self):
        dset = xr.Dataset(
            coords=dict(x=[0, 5, 10, 15]),
            data_vars=dict(w=xr.Variable('x', [100, 100, 10, 10]))
        )
        return dset

    def test_works_with_in_memory_datasets(self, infile, outfile, gridfile):
        conf = dict(
            bins=dict(x=[0, 10, 20]),
            grid=[dict(file=gridfile, variable='w', method='bin')],
            weights='w',
            infile=infile,
            outfile=outfile,
        )

        script.run_conf(conf)

        assert outfile.variables['histogram'][:].tolist() == [100, 20]
