import subprocess
from ladim_aggregate.script import SCRIPT_NAME
from ladim_aggregate import script
import pytest
import subprocess


class Test_run:
    def test_prints_help_message_when_no_arguments(self, capsys):
        script.main2()
        out = capsys.readouterr().out
        assert out.startswith('usage: ' + SCRIPT_NAME)

    def test_prints_help_message_when_help_argument(self, capsys):
        with pytest.raises(SystemExit):
            script.main2('--help')
        out = capsys.readouterr().out
        assert out.startswith('usage: ' + SCRIPT_NAME)


class Test_main:
    def test_can_extract_and_run_example(self, tmp_path):
        import os
        os.chdir(tmp_path)
        r = subprocess.run([SCRIPT_NAME, '--example', 'grid_2D'], stdout=subprocess.PIPE)
        assert r.stdout.decode('utf-8') == ''
        # script.main('--example', 'complex')
        files = {f.name for f in tmp_path.glob('*')}
        assert 'aggregate.yaml' in files
        assert 'count.nc' in files
