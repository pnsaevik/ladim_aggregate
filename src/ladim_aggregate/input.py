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
        self.new_variables = dict()
        self.special_variables = dict()
        self.init_variables = []

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

    def assign(self, **mapping):
        """
        Define new variables from expressions based on the old variables
        :param mapping: A mapping of variable names to functional expressions
        """
        for k, v in mapping.items():
            self._assign(k, v)

    def add_special_variable(self, varname, operator):
        """
        Special variables are the result of aggregation operations such as `max` or
        `min`. They often require a scanning of the whole dataset.

        :param varname: Name of the variable
        :param operator: Name of the operator
        :return: None
        """
        key = operator.upper() + '_' + varname
        self.special_variables[key] = dict(
            key=key,
            varname=varname,
            operator=operator,
            value=None,
        )
        return key

    def special_value(self, key):
        if self.special_variables[key]['value'] is None:
            self._update_special_variables()
        return self.special_variables[key]['value']

    def _assign(self, varname, expression):
        self.new_variables[varname] = create_newvar(expression)

    def _update_special_variables(self):
        # Find all unassigned aggfuncs and store them variable-wise
        spec_keys = [k for k, v in self.special_variables.items() if v['value'] is None]
        spec = {}
        for k in spec_keys:
            opname = self.special_variables[k]['operator']
            vname = self.special_variables[k]['varname']
            spec[vname] = spec.get(vname, []) + [opname]

        if spec == {}:
            return

        out = self.scan(spec)

        for k in spec_keys:
            opname = self.special_variables[k]['operator']
            vname = self.special_variables[k]['varname']
            value = out[vname][opname]
            if opname in ['max', 'min']:
                xr_var = xr.Variable((), value)
            elif opname == 'unique':
                xr_var = xr.Variable(k, value)
            elif opname in ['init', 'final']:
                data, mask = value
                xr_var = xr.Variable('particle', data)
            else:
                raise NotImplementedError
            self.special_variables[k]['value'] = xr_var

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
                    # The "pid aggfuncs" require a special type of input data
                    if fun in ['init', 'final']:
                        data = (data, ddset.variables['pid'].values)
                    out[varname][fun] = update_agg(out[varname][fun], fun, data)
                    agg_log(fun, out[varname][fun])

        def initialize_pid_aggfuncs(num_particles, out_dict):
            # The "pid_aggfuncs" ('init' and 'final') both require a special type of
            # initialization, where the total number of particles must be known.

            # If there are no pid_aggfuncs, abort function
            if not [f for v in out_dict for f in out_dict[v] if f in ['init', 'final']]:
                return

            dtype = np.float32
            fill_value = np.nan
            init_data = np.empty(num_particles, dtype=dtype)
            init_data[:] = fill_value
            init_mask = np.zeros(num_particles, dtype=bool)
            for varname in out_dict:
                for fun in out_dict[varname]:
                    if fun in ['init', 'final']:
                        out_dict[varname][fun] = (init_data, init_mask)

        # Particle variables do only need the first dataset
        with _open_spec(self.datasets[0]) as dset:
            initialize_pid_aggfuncs(dset.dims.get('particle', 0), out)
            update_output(dset, spec)

        spec_without_particle_vars = {
            k: v for k, v in spec.items() if self.datasets[0][k].dims != ('particle', )}

        if spec_without_particle_vars:
            for dset in self.datasets[1:]:
                update_output(dset, spec_without_particle_vars)

        return out

    def chunks(self, filters=None) -> typing.Iterator[xr.Dataset]:
        """
        Return one ladim timestep at a time.

        Variables associated with "time" or "particle" are distributed correctly over
        the particles present in the current timestep.

        For each time step, an optional filter is applied. Furthermore, the derived
        variables added to the dataset are computed.

        :param filters: A filtering expression
        :return: An xarray dataset indexed by "pid" for each time step.
        """
        filterfn = create_newvar(filters)

        # Initialize the "init variables"
        init_variables = {k: None for k in self.init_variables}

        for chunk in ladim_iterator(self.datasets):
            # Apply filter
            filter_idx = None
            num_unfiltered = chunk.dims['pid']
            if filterfn:
                logger.info("Apply filter")
                filter_idx = filterfn(chunk).values
                num_unfiltered = np.count_nonzero(filter_idx)
                logger.info(f'Number of remaining particles: {num_unfiltered}')

            if (num_unfiltered == 0) and (len(init_variables) == 0):
                continue

            # Add derived variables (such as weights and geotags)
            for varname, fn in self.new_variables.items():
                logger.info(f'Compute "{varname}"')
                chunk = chunk.assign(**{varname: fn(chunk)})

            # Add init variables (such as region_INIT)
            for varname, data_and_mask in init_variables.items():
                pid = chunk['pid'].values
                input_data = (chunk[varname].values, pid)
                data, mask = update_init(data_and_mask, input_data)
                init_variables[varname] = (data, mask)
                xr_var = xr.Variable('pid', data[pid])
                chunk = chunk.assign(**{f"{varname}_INIT": xr_var})

            # Add aggregation variables (such as MAX_temp)
            for varname in self.special_variables:
                xr_var = self.special_value(varname)
                chunk = chunk.assign(**{varname: xr_var})

            # Do actual filtering
            if filter_idx is not None:
                chunk = chunk.isel(pid=filter_idx)

            yield chunk


def get_time(timevar):
    return xr.decode_cf(timevar.to_dataset(name='timevar')).timevar.values


def get_newvar_func_from_numexpr(spec):
    import numexpr
    ex = numexpr.NumExpr(spec)

    def weight_fn(chunk):
        args = []
        for n in ex.input_names:
            logger.info(f'Load variable "{n}"')
            args.append(chunk[n].values)
        logger.info(f'Compute expression "{spec}"')
        return xr.Variable('pid', ex.run(*args))

    return weight_fn


def get_newvar_func_from_callable(fn):
    import inspect
    signature = inspect.signature(fn)

    def weight_fn(chunk):
        args = [chunk[n].values for n in signature.parameters.keys()]
        return xr.Variable('pid', fn(*args))

    return weight_fn


def get_newvar_func_from_funcstring(s: str):
    import importlib
    module_name, func_name = s.rsplit('.', 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return get_newvar_func_from_callable(func)


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
    funcs = dict(
        max=update_max, min=update_min, unique=update_unique, init=update_init,
        final=update_final,
    )
    return funcs[aggfun](old, data)


def update_init(old, data, final=False):
    if old is None:
        old = (np.zeros(0, dtype=data[0].dtype), np.zeros(0, dtype=bool))

    # Unpack input arguments
    old_data, old_mask = old
    new_data, new_pid = data

    # Expand array if necessary
    max_pid = np.max(new_pid)
    if max_pid >= len(old_data):
        old_data2 = np.zeros(max_pid + 1, dtype=old_data.dtype)
        old_data2[:len(old_data)] = old_data
        old_mask2 = np.zeros(max_pid + 1, dtype=bool)
        old_mask2[:len(old_mask)] = old_mask
        old_data = old_data2
        old_mask = old_mask2

    if not final:
        # Filter out the new particles
        # Flip because the least recent pid number should be used
        is_unexisting_pid = ~old_mask[new_pid]
        new_data = np.flip(new_data[is_unexisting_pid])
        new_pid = np.flip(new_pid[is_unexisting_pid])

    # Store the new particles
    old_data[new_pid] = new_data
    old_mask[new_pid] = True

    return old_data, old_mask


def update_final(old, data):
    return update_init(old, data, final=True)


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


def create_newvar(spec):
    if spec is None:
        return None
    elif isinstance(spec, tuple) and spec[0] == 'geotag':
        from .geotag import create_geotagger
        return create_geotagger(**spec[1])
    elif isinstance(spec, str):
        if '.' in spec:
            return get_newvar_func_from_funcstring(spec)
        else:
            return get_newvar_func_from_numexpr(spec)
    elif callable(spec):
        return get_newvar_func_from_callable(spec)
    else:
        raise TypeError(f'Unknown type: {type(spec)}')
