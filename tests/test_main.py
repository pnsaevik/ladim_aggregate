from ladim_aggregate import run
import netCDF4 as nc
from uuid import uuid4


class Test_run:
    def test_accepts_minimal_config(self):
        with nc.Dataset(uuid4(), 'w', diskless=True) as outfile:
            run(output_file=outfile)
