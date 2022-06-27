import contextlib
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

        self._attributes = None

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

    @property
    def attributes(self):
        if self._attributes is None:
            spec = self.datasets[0]
            self._attributes = dict()
            with _open_spec(spec) as (dset, _):
                for k, v in dset.variables.items():
                    self._attributes[k] = v.attrs
        return self._attributes

    def _reset_ladim_iterator(self):
        if self._dataset_mustclose:
            self._dataset_current.close()

        def dataset_iterator():
            for spec in self.datasets:
                with _open_spec(spec) as (dset, is_file):
                    self._dataset_mustclose = is_file
                    self._dataset_current = dset
                    yield dset

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

    def scan(self, spec):
        """
        Scan the dataset and return summary values for chosen variables.

        The input parameter `spec` is used to specify summary functions. Each variable
        name can be mapped to one or more functions.

        :param spec: A mapping of variable names to summary function names.
        :returns: A mapping of variable names to summary function name/value pairs.
        """
        out = {k: {fun: None for fun in funclist} for k, funclist in spec.items()}

        def agg_log(aggfunc, aggval):
            if aggfunc == "unique":
                logger.info(f'Number of unique values: {len(aggval)}')
            elif aggfunc == "max":
                logger.info(f'Max value: {aggval}')
            elif aggfunc == "min":
                logger.info(f'Min value: {aggval}')

        def update_output(dset, sub_spec):
            for varname, funclist in sub_spec.items():
                logger.info(f'Load "{varname}" values')
                data = dset.variables[varname].values
                for fun in funclist:
                    out[varname][fun] = update_agg(out[varname][fun], fun, data)
                    agg_log(fun, out[varname][fun])

        # Particle variables do only need the first dataset
        dataset_iterator = self.idatasets()
        first_dset = next(dataset_iterator)
        update_output(first_dset, spec)

        spec_without_particle_vars = {
            k: v for k, v in spec.items() if first_dset[k].dims != ('particle', )}

        if spec_without_particle_vars:
            for next_dset in dataset_iterator:
                update_output(next_dset, spec_without_particle_vars)

        return out

    def idatasets(self) -> typing.Iterator:
        for spec in self.datasets:
            with _open_spec(spec) as (dset, _):
                yield dset

    def read(self):
        try:
            chunk = next(self.ladim_iter)
            logger.info("Apply filter")
            chunk = self.filter(chunk)
            num_unfiltered = chunk.dims['particle_instance']
            logger.info(f'Number of remaining particles: {num_unfiltered}')
            if self.weights and num_unfiltered:
                logger.info("Apply weights")
                chunk = chunk.assign(weights=self.weights(chunk))
            return chunk
        except StopIteration:
            return None

    def chunks(self) -> typing.Iterator[xr.Dataset]:
        chunk = self.read()
        while chunk is not None:
            yield chunk
            chunk = self.read()


def get_time(timevar):
    return xr.decode_cf(timevar.to_dataset(name='timevar')).timevar.values


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
        args = []
        for n in ex.input_names:
            logger.info(f'Load variable "{n}"')
            args.append(chunk[n].values)
        logger.info(f'Compute weights expression "{spec}"')
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
            timestr = str(get_time(dset.time[tidx]).astype('datetime64[s]')).replace("T", " ")
            logger.info(f'Read time step {timestr} (time={dset.time[tidx].values.item()})')
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


def update_agg(old, aggfun, data):
    funcs = dict(max=update_max, min=update_min, unique=update_unique)
    return funcs[aggfun](old, data)


def update_max(old, data):
    if old is None:
        return np.max(data)
    else:
        return max(np.max(data), old)


def update_min(old, data):
    if old is None:
        return np.min(data)
    else:
        return min(np.min(data), old)


def update_unique(old, data):
    if old is None:
        return np.unique(data).tolist()
    else:
        unq = np.unique(data)
        return np.union1d(old, unq).tolist()


@contextlib.contextmanager
def _open_spec(spec):
    if isinstance(spec, str):
        logger.info(f'Open dataset "{spec}"')
        with xr.open_dataset(spec, decode_cf=False) as ddset:
            yield ddset, True
            logger.info(f'Close dataset "{spec}"')
    else:
        logger.info(f'Enter new dataset')
        yield spec, False
