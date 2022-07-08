import subprocess
from ladim_aggregate.script import SCRIPT_NAME
from ladim_aggregate import script, examples
import pytest


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
