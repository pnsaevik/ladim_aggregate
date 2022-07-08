import subprocess
from ladim_aggregate.script import SCRIPT_NAME


class Test_script:
    def test_prints_help_message_when_no_arguments(self):
        r = subprocess.run(SCRIPT_NAME, stdout=subprocess.PIPE)
        assert r.returncode == 0
        assert r.stdout.decode('utf-8').startswith('usage: ' + SCRIPT_NAME)

    def test_prints_help_message_when_help_argument(self):
        r = subprocess.run(SCRIPT_NAME + ' --help', stdout=subprocess.PIPE)
        assert r.returncode == 0
        assert r.stdout.decode('utf-8').startswith('usage: ' + SCRIPT_NAME)
