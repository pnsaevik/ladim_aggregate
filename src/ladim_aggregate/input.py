import contextlib
import logging
import glob
import numpy as np
import xarray as xr
import typing


logger = logging.getLogger(__name__)


class LadimInputStream:
    def __init__(self, spec):
        self.datasets = glob_files(spec)
        self._attributes = None

    @property
    def attributes(self):
        if self._attributes is None:
            spec = self.datasets[0]
            self._attributes = dict()
            with _open_spec(spec) as dset:
                logger.info('Read attributes')
                for k, v in dset.variables.items():
                    self._attributes[k] = v.attrs
        return self._attributes

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

        def update_output(ddset, sub_spec):
            for varname, funclist in sub_spec.items():
                logger.info(f'Load "{varname}" values')
                data = ddset.variables[varname].values
                for fun in funclist:
                    out[varname][fun] = update_agg(out[varname][fun], fun, data)
                    agg_log(fun, out[varname][fun])

        # Particle variables do only need the first dataset
        with _open_spec(self.datasets[0]) as dset:
            update_output(dset, spec)

        spec_without_particle_vars = {
            k: v for k, v in spec.items() if self.datasets[0][k].dims != ('particle', )}

        if spec_without_particle_vars:
            for dset in self.datasets[1:]:
                update_output(dset, spec_without_particle_vars)

        return out

    def chunks(self, filters=None, newvars=None) -> typing.Iterator[xr.Dataset]:
        newvars = newvars or dict()
        filterfn = create_filter(filters)
        newvarfn = {k: create_weights(v) for k, v in newvars.items()}

        for chunk in ladim_iterator(self.datasets):
            num_unfiltered = chunk.dims['pid']
            if filterfn:
                logger.info("Apply filter")
                chunk = filterfn(chunk)
                num_unfiltered = chunk.dims['pid']
                logger.info(f'Number of remaining particles: {num_unfiltered}')

            if newvarfn and num_unfiltered:
                for varname, fn in newvarfn.items():
                    logger.info(f'Compute "{varname}"')
                    chunk = chunk.assign(**{varname: fn(chunk)})

            yield chunk


def get_time(timevar):
    return xr.decode_cf(timevar.to_dataset(name='timevar')).timevar.values


def get_filter_func_from_numexpr(spec):
    import numexpr
    ex = numexpr.NumExpr(spec)

    def filter_fn(chunk):
        args = [chunk[n].values for n in ex.input_names]
        idx = ex.run(*args)
        return chunk.isel(pid=idx)
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
        return xr.Variable('pid', ex.run(*args))

    return weight_fn


def get_filter_func_from_callable(fn):
    import inspect
    signature = inspect.signature(fn)

    def filter_fn(chunk):
        args = [chunk[n].values for n in signature.parameters.keys()]
        idx = fn(*args)
        return chunk.isel(pid=idx)

    return filter_fn


def get_weight_func_from_callable(fn):
    import inspect
    signature = inspect.signature(fn)

    def weight_fn(chunk):
        args = [chunk[n].values for n in signature.parameters.keys()]
        return xr.Variable('pid', fn(*args))

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


def glob_files(spec):
    """
    Convert a set of glob patterns to a list of files

    :param spec: One or more glob patterns
    :return: A list of files
    """

    # Convert input spec to a sequence
    if isinstance(spec, tuple) or isinstance(spec, list):
        specs = spec
    else:
        specs = [spec]

    # Expand glob patterns in spec
    files = []
    for s in specs:
        if isinstance(s, str):
            files += sorted(glob.glob(s))
        else:
            files.append(s)

    return files


def dset_iterator(specs):
    for spec in specs:
        with _open_spec(spec) as dset:
            yield dset


def ladim_iterator(ladim_dsets):
    for dset in dset_iterator(ladim_dsets):
        instance_offset = dset.get('instance_offset', 0)
        pcount_cum = np.concatenate([[0], np.cumsum(dset.particle_count.values)])

        for tidx in range(dset.dims['time']):
            timestr = str(get_time(dset.time[tidx]).astype('datetime64[s]')).replace("T", " ")
            logger.info(f'Read time step {timestr} (time={dset.time[tidx].values.item()})')
            iidx = slice(pcount_cum[tidx], pcount_cum[tidx + 1])
            logger.info(f'Number of particles: {iidx.stop - iidx.start}')
            if iidx.stop == iidx.start:
                continue

            pid = xr.Variable('pid', dset.pid[iidx].values, dset.pid.attrs)

            ddset = xr.Dataset(
                data_vars=dict(instance_offset=instance_offset + iidx.start),
                coords=dict({pid.dims[0]: pid}),
                attrs=dset.attrs,
            )

            for k, v in dset.variables.items():
                if k in ('pid', 'instance_offset'):
                    continue

                logger.info(f'Load variable "{k}"')
                if v.dims == ('particle_instance', ):
                    new_var = xr.Variable(pid.dims[0], v[iidx].values, v.attrs)
                elif v.dims == ('particle', ):
                    new_var = xr.Variable(pid.dims[0], v.values[pid.values], v.attrs)
                elif v.dims == ('time', ):
                    data = np.broadcast_to(v.values[tidx], (iidx.stop - iidx.start, ))
                    new_var = xr.Variable(pid.dims[0], data, v.attrs)
                else:
                    raise ValueError(f'Unknkown dimension: "{v.dims}"')

                ddset = ddset.assign(**{k: new_var})

            yield ddset


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
            yield ddset
            logger.info(f'Close dataset "{spec}"')
    else:
        logger.info(f'Enter new dataset')
        yield spec


def create_filter(spec):
    if spec is None:
        return None
    elif isinstance(spec, str):
        if '.' in spec:
            return get_filter_func_from_funcstring(spec)
        else:
            return get_filter_func_from_numexpr(spec)
    elif callable(spec):
        return get_filter_func_from_callable(spec)
    else:
        raise TypeError(f'Unknown type: {type(spec)}')


def create_weights(spec):
    if spec is None:
        return None
    elif isinstance(spec, str):
        if '.' in spec:
            return get_weight_func_from_funcstring(spec)
        else:
            return get_weight_func_from_numexpr(spec)
    elif callable(spec):
        return get_weight_func_from_callable(spec)
    else:
        raise TypeError(f'Unknown type: {type(spec)}')
