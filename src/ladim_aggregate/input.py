import contextlib
import logging
import glob
import numpy as np
import xarray as xr
import typing
import re


logger = logging.getLogger(__name__)


class LadimInputStream:
    def __init__(self, spec):
        self.datasets = glob_files(spec)
        self._attributes = None
        self._derived_variables = dict()
        self._agg_variables = dict()
        self._init_variables = []

    @property
    def attributes(self):
        if self._attributes is None:
            self._attributes = dict()
            with self.open_dataset(0) as dset:
                logger.debug('Read attributes')
                for k, v in dset.variables.items():
                    self._attributes[k] = v.attrs
        return self._attributes

    def open_dataset(self, idx: int) -> xr.Dataset:
        """
        Open ladim dataset by index.

        :param idx: Ordinal index of the dataset
        :return: The dataset
        """
        return _open_spec(self.datasets[idx])

    def add_derived_variable(self, varname, definition):
        """
        Define new variables from expressions based on the old variables
        :param varname: Variable name
        :param definition: Variable definition
        """
        self._derived_variables[varname] = create_varfunc(definition)

    def add_aggregation_variable(self, varname, operator):
        """
        Aggregation variables are the result of aggregation operations such as `max` or
        `min`. They require a scanning of the whole dataset. The scanning is deferred
        until a specific aggregate value is requested.

        Variables defined here are available through the function get_aggregation_value

        :param varname: Name of the variable
        :param operator: Name of the operator
        :return: The name of the aggregate variable (e.g., MAX_temp)
        """
        key = operator.upper() + '_' + varname
        self._agg_variables[key] = dict(
            key=key,
            varname=varname,
            operator=operator,
            value=None,
        )
        return key

    def get_aggregation_value(self, key):
        if self._agg_variables[key]['value'] is None:
            self._update_agg_variables()
        return self._agg_variables[key]['value']

    def add_init_variable(self, varname):
        """
        Init variables are derived variables that do not need a pre-scanning of the
        dataset. The value of an init variable is the first value found when looping
        through the time steps.

        Variables added here are available through the chunks() function as variables
        named like `<varname>_INIT`.

        :param varname: Variable name
        :return: None
        """
        self._init_variables.append(varname)

    def add_grid_variable(self, data_array, method):
        """
        Grid variables are derived variables that are interpolated from a grid. They
        are added to the dataset through the chunks() function.

        :param data_array: A named xarray DataArray defining the variable values on a grid
        :param method: Either 'linear', 'nearest' or 'bin'
        :return: None
        """
        self._derived_variables[data_array.name] = create_varfunc(('grid', data_array, method))

    def _update_agg_variables(self):
        # Find all unassigned aggfuncs and store them variable-wise
        spec_keys = [k for k, v in self._agg_variables.items() if v['value'] is None]
        spec = {}
        for k in spec_keys:
            opname = self._agg_variables[k]['operator']
            vname = self._agg_variables[k]['varname']
            spec[vname] = spec.get(vname, []) + [opname]

        if spec == {}:
            return

        out = self.scan(spec)

        for k in spec_keys:
            opname = self._agg_variables[k]['operator']
            vname = self._agg_variables[k]['varname']
            value = out[vname][opname]
            if opname in ['max', 'min']:
                xr_var = xr.Variable((), value)
            elif opname == 'unique':
                xr_var = xr.Variable(k, value)
            else:
                raise NotImplementedError
            self._agg_variables[k]['value'] = xr_var

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
                logger.debug(f'Number of unique values: {len(aggval)}')
            elif aggfunc == "max":
                logger.debug(f'Max value: {aggval}')
            elif aggfunc == "min":
                logger.debug(f'Min value: {aggval}')

        def update_output(ddset, sub_spec):
            for varname, funclist in sub_spec.items():
                if varname in ddset.variables:
                    logger.debug(f'Load "{varname}" values')
                    data = ddset.variables[varname].values
                elif varname in self._derived_variables:
                    logger.debug(f'Compute "{varname}" values')
                    fn = self._derived_variables[varname]
                    data = fn(ddset).values
                else:
                    raise ValueError(f'Unknown variable name: "{varname}"')

                for fun in funclist:
                    out[varname][fun] = update_agg(out[varname][fun], fun, data)
                    agg_log(fun, out[varname][fun])

        # Particle variables do only need the first dataset
        with self.open_dataset(0) as dset:
            update_output(dset, spec)
            pvars = [k for k in spec if k in dset and dset[k].dims == ('particle', )]
            spec_without_particle_vars = {k: v for k, v in spec.items() if k not in pvars}

        if spec_without_particle_vars:
            for idx in range(1, len(self.datasets)):
                with self.open_dataset(idx) as dset:
                    update_output(dset, spec_without_particle_vars)

        return out

    def chunks(self, filters=None, timestep_filter=None, particle_filter=None) -> typing.Iterator[xr.Dataset]:
        """
        Return one ladim timestep at a time.

        Variables associated with "time" or "particle" are distributed correctly over
        the particles present in the current timestep.

        For each time step, an optional filter is applied. Furthermore, the derived
        variables added to the dataset are computed.

        :param filters: A filtering expression
        :param timestep_filter: A set of time steps indices to return. If None (default),
        return all timesteps. The time steps list is sorted before being applied, and
        nonunique elements are disregarded.
        :param particle_filter: The same as `filter`, except that it only returns a
        single instance per particle (i.e., the first occurrence when the condition is
        met).
        :return: An xarray dataset indexed by "pid" for each time step.
        """
        filterfn = create_varfunc(filters)

        if particle_filter is not None:
            particle_filterfn = create_varfunc(('pfilter', particle_filter))
        else:
            particle_filterfn = None

        # Initialize the "init variables"
        init_variables = {k: None for k in self._init_variables}

        for chunk in ladim_iterator(self.datasets, timestep_filter):
            # Apply filter
            filter_idx = None
            num_unfiltered = chunk.sizes['pid']
            if filterfn:
                logger.debug("Apply filter")
                filter_idx = filterfn(chunk).values
                num_unfiltered = np.count_nonzero(filter_idx)
                logger.debug(f'Number of remaining particles: {num_unfiltered}')

            if (num_unfiltered == 0) and (len(init_variables) == 0):
                continue

            # Add derived variables (such as weights and geotags)
            for varname, fn in self._derived_variables.items():
                logger.debug(f'Compute "{varname}"')
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
            for varname in self._agg_variables:
                xr_var = self.get_aggregation_value(varname)
                chunk = chunk.assign(**{varname: xr_var})

            # Add particle filtering
            if particle_filterfn is not None:
                logger.debug("Apply particle filter")
                pfilter_idx = particle_filterfn(chunk).values
                if filterfn:
                    filter_idx = filter_idx & pfilter_idx
                else:
                    filter_idx = pfilter_idx
                num_unfiltered = np.count_nonzero(filter_idx)
                logger.debug(f'Number of remaining particles: {num_unfiltered}')

            # Do actual filtering
            if filter_idx is not None:
                chunk = chunk.isel(pid=filter_idx)

            yield chunk


def get_time(timevar):
    return xr.decode_cf(timevar.to_dataset(name='timevar')).timevar.values


def get_varfunc_from_numexpr(spec):
    import numexpr
    ex = numexpr.NumExpr(spec)

    def weight_fn(chunk):
        args = []
        for n in ex.input_names:
            logger.debug(f'Load variable "{n}"')
            args.append(chunk[n].values)
        logger.debug(f'Compute expression "{spec}"')
        out_shape = chunk['pid'].shape
        result = ex.run(*args)  # Might be single-valued
        result_reshaped = np.broadcast_to(result, out_shape)
        return xr.Variable('pid', result_reshaped)

    return weight_fn


def get_varfunc_from_callable(fn):
    import inspect
    signature = inspect.signature(fn)

    def weight_fn(chunk):
        args = []
        for name, param in signature.parameters.items():
            if name in chunk:
                args.append(chunk[name].values)
            elif param.default == inspect.Parameter.empty:
                raise KeyError(f'Variable {name} not in dataset (required by {fn})')
            else:
                args.append(param.default)

        return xr.Variable('pid', fn(*args))

    return weight_fn


def get_varfunc_from_funcstring(s: str):
    import importlib
    module_name, func_name = s.rsplit('.', 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return get_varfunc_from_callable(func)


def get_varfunc_from_grid(darr: xr.DataArray, method):
    def fn_other(chunk):
        coords = {d: chunk.variables[d] for d in darr.dims}
        return darr.interp(coords, method=method).variable

    def fn_bin(chunk):
        idx = {}
        for d in darr.dims:
            dvalues = darr[d].values
            pvalues = chunk.variables[d].values
            data = np.searchsorted(dvalues, pvalues, side='right') - 1
            idx[d] = xr.Variable(dims='pid', data=data)
        return darr.isel(idx).variable

    if method == 'bin':
        return fn_bin
    else:
        return fn_other


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


def ladim_iterator(ladim_dsets, timesteps=None):
    """
    Return one chunk of a ladim dataset per timestep

    Distributes particle and time variable over the `particle_instance` dimension.

    :param ladim_dsets: A sequence of ladim datasets
    :param timesteps: A set of time steps indices to return. If None (default), return
    all timesteps. The time steps list is sorted before being applied, and nonunique
    elements are disregarded.
    :return: An iterator over timestep chunks
    """

    # Prepare to filter out some time steps
    tidx_start = 0
    if timesteps is not None:
        timesteps = np.unique(timesteps)

    for dset in dset_iterator(ladim_dsets):
        instance_offset = dset.get('instance_offset', 0)
        pcount_cum = np.concatenate([[0], np.cumsum(dset.particle_count.values)])
        tidx_list = range(dset.sizes['time'])

        # Potentially filter out some time steps
        if timesteps is not None:
            include_steps = timesteps - tidx_start
            tidx_list = np.intersect1d(tidx_list, include_steps)
            tidx_start += dset.sizes['time']

        for tidx in tidx_list:
            timestr = str(get_time(dset.time[tidx]).astype('datetime64[s]')).replace("T", " ")
            logger.info(f'Read time step {timestr} (time={dset.time[tidx].values.item()})')
            iidx = slice(pcount_cum[tidx], pcount_cum[tidx + 1])
            logger.debug(f'Number of particles: {iidx.stop - iidx.start}')
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

                logger.debug(f'Load variable "{k}"')
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
        logger.debug(f'Open dataset "{spec}"')
        with xr.open_dataset(spec, decode_cf=False) as ddset:
            yield ddset
            logger.debug(f'Close dataset "{spec}"')
    else:
        logger.debug(f'Enter new dataset')
        yield spec


def create_varfunc(spec):
    if spec is None:
        return None
    elif isinstance(spec, tuple) and spec[0] == 'geotag':
        from .geotag import create_geotagger
        return create_geotagger(**spec[1])
    elif isinstance(spec, tuple) and spec[0] == 'pfilter':
        return create_pfilter(spec[1])
    elif isinstance(spec, tuple) and spec[0] == 'grid':
        return get_varfunc_from_grid(spec[1], spec[2])
    elif isinstance(spec, str):
        if re.match(r'[.\w]+\.\w+$', spec):
            return get_varfunc_from_funcstring(spec)
        else:
            return get_varfunc_from_numexpr(spec)
    elif isinstance(spec, int) or isinstance(spec, float):
        return get_varfunc_from_numexpr(str(spec))
    elif callable(spec):
        return get_varfunc_from_callable(spec)
    else:
        raise TypeError(f'Unknown type: {type(spec)}')


def create_pfilter(spec):
    fn = create_varfunc(spec)
    has_been_triggered = ResizableArray(100, dtype=bool)

    def pfilter(chunk):
        fn_val = fn(chunk)
        pid = chunk['pid'].values
        max_pid = pid.max()
        if max_pid >= len(has_been_triggered):
            has_been_triggered.resize(max_pid + 1)
        is_new = ~has_been_triggered.data[pid]
        condition = fn_val & is_new
        pid_triggered = pid[condition]
        has_been_triggered.data[pid_triggered] = True
        return xr.Variable('pid', condition)

    return pfilter


class ResizableArray:
    def __init__(self, capacity, dtype):
        self.data = np.zeros(capacity, dtype=dtype)

    def resize(self, new_capacity):
        old_data = self.data
        self.data = np.zeros(new_capacity, dtype=old_data.dtype)
        self.data[:len(old_data)] = old_data

    def __len__(self):
        return len(self.data)
