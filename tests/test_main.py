from ladim_aggregate import run
from uuid import uuid4


class Test_run:
    def test_accepts_minimal_config(self):
        run(output_file=uuid4(), diskless=True)
