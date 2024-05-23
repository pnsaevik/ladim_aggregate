class Example:
    def __init__(self, name):
        self.name = name
        self._files = None
        self._config = None

    @staticmethod
    def available():
        import pkgutil
        return [m.name for m in pkgutil.iter_modules(__path__) if m.ispkg]

    @property
    def package(self):
        return __name__ + '.' + self.name

    @property
    def config(self):
        if self._config is None:
            import yaml
            data = self.files()['aggregate.yaml']
            self._config = yaml.safe_load(data.decode('utf-8'))
        return self._config

    def files(self, glob=None):
        if glob is None:
            return self.data

        import re
        re_pattern = '^' + glob.replace('.', '\\.').replace('?', '.').replace('*', '.*') + '$'
        return {k: v for k, v in self.data.items() if re.match(re_pattern, k)}

    @property
    def descr(self):
        import re
        import pkgutil
        data = pkgutil.get_data(self.package, 'aggregate.yaml')
        config_txt = data.decode(encoding='utf-8')
        match = re.match('^# (.*)', config_txt)
        if match:
            return match.group(1)
        else:
            return ""

    @property
    def data(self):
        if self._files is None:
            self._files = self._load()
        return self._files

    def _load(self):
        import importlib.resources

        data = dict()

        for file in importlib.resources.files(self.package).iterdir():
            fname = file.name
            if fname.startswith('__'):
                continue

            data[fname] = file.read_bytes()

            if fname.endswith('.nc.yaml'):
                import yaml
                import xarray as xr
                key = fname[:-5]
                yaml_text = data[fname].decode(encoding='utf-8')
                xr_dict = yaml.safe_load(yaml_text)
                xr_dset = xr.Dataset.from_dict(xr_dict)
                data[key] = xr_dset

        return data

    def extract(self, dirname='.'):
        import logging
        from pathlib import Path
        import xarray as xr
        logger = logging.getLogger(__name__)
        outdir = Path(dirname)

        output_files = [k for k in self.files(self.config['outfile'])]
        input_files = {k: v for k, v in self.files().items() if k not in output_files}

        for fname, data in input_files.items():
            if fname.endswith('.nc.yaml'):
                continue

            logger.info(f'Extract input file: "{fname}"')
            path = outdir / fname
            if fname.endswith('.nc'):
                assert isinstance(data, xr.Dataset)
                data.to_netcdf(path)
            else:
                with open(path, 'bw') as f:
                    f.write(data)

        return 'aggregate.yaml'

    def run(self):
        from .. import script
        from ..output import MultiDataset
        from ..input import LadimInputStream

        ladim_dsets = list(self.files(self.config['infile']).values())
        ladim_input_stream = LadimInputStream(ladim_dsets)

        outfile_name = self.config['outfile']
        with MultiDataset(outfile_name, diskless=True) as output_dset:
            script.run(ladim_input_stream, self.config, output_dset, self.files())
            result = output_dset.to_dict()

        import yaml
        out_pattern = self.config['outfile'].replace('.nc', '*.nc.yaml')
        expected = {
            k[:-5]: yaml.safe_load(v.decode(encoding='utf-8'))
            for k, v in self.files(out_pattern).items()
        }

        return result, expected


def nc_dump(dset):
    """Returns the contents of an open netCDF4 dataset as a dict"""

    variables = dict()
    for name in dset.variables:
        v = dict()
        v['dims'] = list(dset.variables[name].dimensions)
        if len(v['dims']) == 1:
            v['dims'] = v['dims'][0]

        v['data'] = dset.variables[name][:].tolist()

        atts = dict()
        for attname in dset.variables[name].ncattrs():
            atts[attname] = dset.variables[name].getncattr(attname)
        if atts:
            v['attrs'] = atts

        variables[name] = v

    return variables
