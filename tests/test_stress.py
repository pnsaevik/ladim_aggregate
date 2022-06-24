from ladim_aggregate.script import main
from pathlib import Path
import yaml


root = Path(r"C:\Users\a5606\AppData\Local\Temp\stresstest")


def test_lice():
    config = dict(
        infile=r"S:\marimy\Lusearkiv_PO5\Raw2018\raw_0000.nc",
        outfile=str(root / "count.nc"),
        bins=dict(time="group_by", Y=1, X=1),
        filter="(40 <= age) & (age < 170) & (Z <= 2)",
        weights="super",
        filesplit_dims=['farmid', 'time'],
    )
    config_fname = str(root / "aggregate.yaml")
    with open(config_fname, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f)

    main(config_fname)


if __name__ == "__main__":
    test_lice()
