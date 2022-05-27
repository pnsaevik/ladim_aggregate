import logging
import glob
import numpy as np
import xarray as xr
import typing


logger = logging.getLogger(__name__)


class LadimInputStream:
    def __init__(self, spec):
        # Convert input spec to a sequence
        if isinstance(spec, tuple) or isinstance(spec, list):
            specs = spec
        else:
            specs = [spec]

        # Expand glob patterns in spec
        self.datasets = []
        for s in specs:
            if isinstance(s, str):
                self.datasets += sorted(glob.glob(s))
            else:
                self.datasets.append(s)
        logger.info(f'Number of input datasets: {len(self.datasets)}')

        self._filter = lambda chunk: chunk
        self._weights = None

        self._dataset_iterator = None
        self._dataset_current = xr.Dataset()
        self._dataset_mustclose = False
        self.ladim_iter = None
        self._reset_ladim_iterator()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._dataset_mustclose:
            self._dataset_current.close()

    def close(self):
        self._dataset_current.close()

    def seek(self, pos):
        if pos != 0:
            raise NotImplementedError
        self._reset_ladim_iterator()

    def _reset_ladim_iterator(self):
        if self._dataset_mustclose:
            self._dataset_current.close()

        def dataset_iterator():
            for spec in self.datasets:
                if isinstance(spec, str):
                    logger.info(f'Open input dataset {spec}')
                    with xr.open_dataset(spec) as dset:
                        logger.info(f'Number of particle instances: {dset.dims["particle_instance"]}')
                        self._dataset_current = dset
                        self._dataset_mustclose = True
                        yield dset
                else:
                    logger.info(f'Enter new dataset')
                    self._dataset_current = spec
                    self._dataset_mustclose = False
                    logger.info(f'Number of particle instances: {spec.dims["particle_instance"]}')
                    yield spec

        self._dataset_iterator = dataset_iterator()
        self.ladim_iter = ladim_iterator(self._dataset_iterator)

    @property
    def filter(self):
        return self._filter

    @property
    def weights(self):
        return self._weights

    @filter.setter
    def filter(self, spec):
        if spec is None:
            return
        elif isinstance(spec, str):
            if '.' in spec:
                self._filter = get_filter_func_from_funcstring(spec)
            else:
                self._filter = get_filter_func_from_numexpr(spec)
        elif callable(spec):
            self._filter = get_filter_func_from_callable(spec)
        else:
            raise TypeError(f'Unknown type: {type(spec)}')

    @weights.setter
    def weights(self, spec):
        if spec is None:
            return
        elif isinstance(spec, str):
            if '.' in spec:
                self._weights = get_weight_func_from_funcstring(spec)
            else:
                self._weights = get_weight_func_from_numexpr(spec)
        elif callable(spec):
            self._weights = get_weight_func_from_callable(spec)
        else:
            raise TypeError(f'Unknown type: {type(spec)}')

    def find_limits(self, resolution):
        def iterate_datasets() -> typing.Iterable:
            for spec in self.datasets:
                if isinstance(spec, str):
                    logger.info(f'Open input dataset {spec}')
                    with xr.open_dataset(spec) as ddset:
                        yield ddset
                else:
                    logger.info(f'Enter new dataset')
                    yield spec

        def t64conv(timedelta_or_other):
            try:
                t64val, t64unit = timedelta_or_other
                return np.timedelta64(t64val, t64unit)
            except TypeError:
                return timedelta_or_other

        # Align to wholenumber resolution
        def align(val_raw, res_raw):
            if np.issubdtype(np.array(res).dtype, np.timedelta64):
                val_posix = (val_raw - np.datetime64('1970-01-01')).astype('timedelta64[us]')
                res_posix = res.astype('timedelta64[us]')
                ret_posix = (val_posix.astype('i8') // res_posix.astype('i8')) * res_posix
                return np.datetime64('1970-01-01') + ret_posix
            else:
                return np.array((val_raw // res_raw) * res_raw).item()

        varnames = resolution.keys()
        logger.info("Limits are not given, compute automatically from input file")
        minvals = {k: [] for k in varnames}
        maxvals = {k: [] for k in varnames}
        for dset in iterate_datasets():
            for k in varnames:
                res = t64conv(resolution[k])
                minval = align(dset.variables[k].min().values, res)
                maxval = align(dset.variables[k].max().values + res, res)
                logger.info(f'Limits for `{k}` in current dataset: [{minval}, {maxval}]')
                minvals[k].append(minval)
                maxvals[k].append(maxval)

        lims = {k: [np.min(minvals[k]), np.max(maxvals[k])] for k in varnames}
        for k in varnames:
            logger.info(f'Final limits for {k}: [{lims[k][0]}, {lims[k][1]}]')
        return lims

    def read(self):
        try:
            chunk = next(self.ladim_iter)
            logger.info("Apply filter")
            chunk = self.filter(chunk)
            num_unfiltered = chunk.dims['particle_instance']
            logger.info(f'Number of unfiltered particles: {num_unfiltered}')
            if self.weights and (num_unfiltered == 0):
                logger.info("Apply weights")
                chunk = chunk.assign(weights=self.weights(chunk))
            return chunk
        except StopIteration:
            return None

    def chunks(self) -> typing.Iterable:
        chunk = self.read()
        while chunk is not None:
            yield chunk
            chunk = self.read()


def get_filter_func_from_numexpr(spec):
    import numexpr
    ex = numexpr.NumExpr(spec)

    def filter_fn(chunk):
        args = [chunk[n].values for n in ex.input_names]
        idx = ex.run(*args)
        return chunk.isel(particle_instance=idx)
    return filter_fn


def get_weight_func_from_numexpr(spec):
    import numexpr
    ex = numexpr.NumExpr(spec)

    def weight_fn(chunk):
        args = [chunk[n].values for n in ex.input_names]
        return xr.Variable('particle_instance', ex.run(*args))

    return weight_fn


def get_filter_func_from_callable(fn):
    import inspect
    signature = inspect.signature(fn)

    def filter_fn(chunk):
        args = [chunk[n].values for n in signature.parameters.keys()]
        idx = fn(*args)
        return chunk.isel(particle_instance=idx)

    return filter_fn


def get_weight_func_from_callable(fn):
    import inspect
    signature = inspect.signature(fn)

    def weight_fn(chunk):
        args = [chunk[n].values for n in signature.parameters.keys()]
        return xr.Variable('particle_instance', fn(*args))

    return weight_fn


def get_filter_func_from_funcstring(s: str):
    import importlib
    module_name, func_name = s.rsplit('.', 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return get_filter_func_from_callable(func)


def get_weight_func_from_funcstring(s: str):
    import importlib
    module_name, func_name = s.rsplit('.', 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return get_weight_func_from_callable(func)


def ladim_iterator(ladim_dsets):
    for dset in ladim_dsets:
        instance_offset = dset.get('instance_offset', 0)
        pcount_cum = np.concatenate([[0], np.cumsum(dset.particle_count.values)])

        for tidx in range(dset.dims['time']):
            logger.info(f'Read time step {dset.time[tidx].values}')
            iidx = slice(pcount_cum[tidx], pcount_cum[tidx + 1])
            logger.info(f'Number of particles: {iidx.stop - iidx.start}')
            if iidx.stop == iidx.start:
                continue
            pidx = xr.Variable('particle_instance', dset.pid[iidx].values)
            ttidx = xr.Variable('particle_instance', np.broadcast_to(tidx, (len(pidx), )))
            ddset = dset.isel(time=ttidx, particle_instance=iidx)
            if 'particle' in dset.dims:
                ddset = ddset.isel(particle=pidx)

            ddset = ddset.assign(instance_offset=instance_offset + iidx.start)
            yield ddset


def _ladim_iterator_read_variable(dset, varname, tidx, iidx, pidx):
    v = dset.variables[varname]
    first_dim = v.dims[0]
    if first_dim == 'particle_instance':
        return v[iidx].values
    elif first_dim == 'particle':
        return v[pidx].values
    elif first_dim == 'time':
        return v[tidx].values
    else:
        raise ValueError(f'Unknown dimension type: {first_dim}')
