import pandas as pd
import numpy as np
import logging
import cftime
import xarray as xr
import datetime
import typing
from .input import LadimInputStream


logger = logging.getLogger(__name__)
BinSpec = typing.Dict[typing.Literal['centers', 'edges'], np.typing.NDArray]


def get_centers_from_edges(edges):
    edgediff = edges[1:] - edges[:-1]
    return edges[:-1] + 0.5 * edgediff


def densify_sparse_histogram(coords, vals):
    """
    Create dense histogram chunk from sparse histogram
    
    :param coords: Bin indices, sequence of vectors, each having size N
    :param vals: Bin weights, one vector of size N
    :return: hist_chunk, idx_slice where hist_chunk is a dense matrix
        containing the bin weights, and idx_slice is a tuple of slices
        representing the position of the histogram chunk within the full
        histogram matrix.
    """
    # Abort if there are no points left
    if len(vals) == 0:
        num_coords = np.shape(coords)[0]
        return np.zeros((0, ) * num_coords), (slice(1, 0), ) * num_coords

    # Find min and max bin edges to be used
    idx = [(np.min(c), np.max(c) + 1) for c in coords]
    idx_slice = tuple(slice(start, stop) for start, stop in idx)

    # Densify
    shifted_coords = coords - np.asarray([start for start, _ in idx])[:, np.newaxis]
    shape = tuple(stop - start for start, stop in idx)
    hist_chunk = np.zeros(shape, dtype=vals.dtype)
    hist_chunk[tuple(shifted_coords)] = vals

    return hist_chunk, idx_slice


def convert_datebins(spec: dict, dset: xr.Dataset) -> dict:
    """
    Convert bin specifications containing dates.

    The function takes a bin specification as input, as well as an xarray.Dataset
    where the data variables may potentially contain cfunits date attributes. If one
    of the bin specifications is specified as a date (either date string or python/numpy
    object), and connected to a variable with date attributes, the bin specification is
    converted to a number according to the variable's cfunits date specification.

    The function works for the following types of bin formats:

    1. Pure list - ['2001-01', np.datetime('2001-01-02T12')]
    2. Label format: dict(edges=['2001-01', '2001-02', '2001-03'], labels=['J', 'F'])
    3. Range format: dict(min='2001-01', max='2001-03', step='1 months')

    Date steps can be specified as 'years', 'months', 'days', 'hours', 'minutes',
    'seconds', or a numpy timedelta64 data type.

    :param spec: A dict where the keys are the variable names and the values are the bin
                 specifications.
    :param dset: A dataset where the variables may have cfunits date attributes
    :return:     A dict of the same format as ``spec``, except that dates are converted
                 to numbers.
    """
    newspec = {}

    # k is the variable name
    # v is the bin format
    for k, v in spec.items():
        if k not in dset:
            attrs = {}
        else:
            attrs = dset[k].attrs
        units = attrs.get('units', '')
        calendar = attrs.get('calendar', 'standard')
        newspec[k] = convert_binspec(v, units, calendar)

    return newspec


def convert_binspec(singlespec, units, calendar):
    if 'since' not in units:
        return singlespec

    elif isinstance(singlespec, list):
        return [convert_date(v, units, calendar) for v in singlespec]

    elif isinstance(singlespec, dict):
        result = {k: v for k, v in singlespec.items()}

        if 'min' in result:
            result['min'] = convert_date(result['min'], units, calendar)

        if 'max' in result:
            result['max'] = convert_date(result['max'], units, calendar)

        if 'edges' in result:
            result['edges'] = [convert_date(e, units, calendar) for e in result['edges']]

        if 'step' in result:
            result['step'] = convert_step(result['step'], units, calendar)

        return result

    elif isinstance(singlespec, str) and singlespec.count(' ') == 1:
        return convert_step(singlespec, units, calendar)

    else:
        return singlespec


def convert_step(step_spec, units, calendar):
    if isinstance(step_spec, str):
        step_str, units_in = step_spec.split(sep=' ', maxsplit=1)
        step = float(step_str)
        units_out, _, refdate = units.split(sep=' ', maxsplit=2)
    else:
        return step_spec

    date_in = cftime.num2date(step, f'{units_in} since {refdate}', calendar)
    return cftime.date2num(date_in, f'{units_out} since {refdate}', calendar)


def convert_date(value, units, calendar):
    """
    Convert the input value to a cfunits date format. The input value can be anything
    convertible to np.datetime64.

    :param value: Input date
    :param units: Output cfunits format
    :param calendar: Output cfunits calendar
    :return: Date in cfunits format
    """
    try:
        value = np.datetime64(value).astype(object)
    except ValueError:
        return value

    if isinstance(value, datetime.date):
        value = datetime.datetime.combine(value, datetime.time())

    return cftime.date2num(value, units, calendar)


def autobins(spec: dict, dset: LadimInputStream | None):
    if dset is not None and hasattr(dset, 'open_dataset'):
        with dset.open_dataset(0) as xr_dset:
            spec = convert_datebins(spec, xr_dset)

    # Add INIT bins, if any
    for k in spec:
        if k.endswith('_INIT') and dset is not None:
            varname = k[:-5]
            dset.add_init_variable(varname)

    # Find bin specification type
    spec_types = dict()
    for k, v in spec.items():
        if isinstance(v, list):
            spec_types[k] = 'edges'

        elif isinstance(v, dict) and all(u in v for u in ['min', 'max', 'step']):
            spec_types[k] = 'range'

        elif isinstance(v, dict) and all(u in v for u in ['edges', 'labels']):
            spec_types[k] = 'edges_labels'

        elif isinstance(v, dict) and 'step' in v:
            spec_types[k] = 'resolution'

        elif v == 'group_by' or v == 'unique':
            spec_types[k] = 'unique'

        elif np.issubdtype(type(v), np.number):
            spec[k] = dict(align=0, step=v)
            spec_types[k] = 'resolution'

        elif isinstance(v, str):
            specsplit = v.split(sep=" ")
            if len(specsplit) == 2:  # Single number with units
                spec_types[k] = 'resolution'
            else:
                raise ValueError(f'Unknown bin type: {v}')

        else:
            raise ValueError(f'Unknown bin type: {v}')

    # Check if we need pre-scanning of the dataset
    scan_params_template = dict(unique=['unique'], resolution=['min', 'max'])
    scan_params = {k: scan_params_template[v] for k, v in spec_types.items()
                   if v in scan_params_template}
    scan_output = {k: dict() for k in spec}
    if scan_params and dset is not None:
        # First add all the variable definitions...
        aggvars = []
        for varname, aggfuncs in scan_params.items():
            for aggfun in aggfuncs:
                if varname.endswith('_INIT'):
                    # If init variable, use the base variable for aggregation
                    aggvar = dset.add_aggregation_variable(varname[:-5], aggfun)
                else:
                    aggvar = dset.add_aggregation_variable(varname, aggfun)
                aggvars.append((aggvar, varname, aggfun))

        logger.info(f'Scan input dataset to find {", ".join([s[0] for s in aggvars])}')

        # ... then trigger the scanning of the dataset
        for aggvar, varname, aggfun in aggvars:
            scan_output[varname][aggfun] = dset.get_aggregation_value(aggvar)

    # Put the specs and the result of the pre-scanning into the bin generator
    bins = {k: bin_generator(spec[k], spec_types[k], scan_output[k]) for k in spec}

    return bins


def bin_generator(
        spec: typing.Any,
        spec_type: str,
        scan_output: dict[str, typing.Any],
    ) -> BinSpec:

    if spec_type == 'edges':
        edges = np.asarray(spec)
        centers = get_centers_from_edges(edges)
    elif spec_type == 'edges_labels':
        edges = np.asarray(spec['edges'])
        centers = np.asarray(spec['labels'])
    elif spec_type == 'range':
        edges = np.arange(spec['min'], spec['max'] + spec['step'], spec['step'])
        centers = get_centers_from_edges(edges)
    elif spec_type == 'unique':
        data = scan_output['unique']
        edges = np.concatenate([data, [data[-1] + 1]])
        centers = np.asarray(data)
    elif spec_type == 'resolution':
        edges = generate_1d_grid(
            start=np.asarray(scan_output['min']),
            stop=np.asarray(scan_output['max']),
            step=np.asarray(spec['step']),
            align=spec.get('align', 0),
        )
        centers = get_centers_from_edges(edges)
    else:
        raise ValueError(f'Unknown spec_type: {spec_type}')

    return {'edges': edges, 'centers': centers}


def t64conv(timedelta_or_other):
    """
    Convert input data to timedelta64.
    
    Acceptable input formats are for instance ``[value, unit]`` and ``"23 hours"``.
    Timedelta input are returned verbatim.
    """
    if isinstance(timedelta_or_other, np.timedelta64):
        return timedelta_or_other
    elif isinstance(timedelta_or_other, str) and timedelta_or_other.count(' ') == 1:
        t64val_str, t64unit_cf = timedelta_or_other.split(sep=" ")
        t64val = int(t64val_str)
        if t64unit_cf.endswith('s') and len(t64unit_cf) > 3:  # Remove plurals
            t64unit_cf = t64unit_cf[:-1]
        unitconv = dict(day='D', hour='h', minute='m', second='s', millisecond='ms',
                        microsecond='us')
        t64unit = unitconv.get(t64unit_cf, t64unit_cf)
        return np.timedelta64(t64val, t64unit)

    try:
        t64val, t64unit = timedelta_or_other
        return np.timedelta64(t64val, t64unit)
    except TypeError:
        raise ValueError(f'Could not convert input value to timedelta64: {timedelta_or_other}')


def generate_1d_grid(start, stop, step, align=None):
    """
    Generate a 1-dimensional grid.

    The returned grid `g` is a numpy array with following properties:

    1.  The grid spacing is determined by `step`.
        That is, `g[i+1] - g[i] = step` for all integers i.

    2.  The values `start` and `stop` are properly contained in the range.
        That is, `g[0] <= start <= stop < g[-1]`

    3.  The grid is aligned with the value `align`.
        That is, `align = start + N * step` for some (possibly negative) int N.

    The grid range might be slightly expanded compared to `start` and `stop`
    in order to achieve this.

    :param step: Grid resolution
    :param start: Min value which should be contained in grid
    :param stop: Max value which should be contained in grid
    :param align: Value which should be aligned with grid
    :return: Initialized grid satisfying all conditions
    """

    # If necessary, move start position to ensure alignment
    if align is not None:
        start = align_start_of_range(start, step, align)

    # Compute number of grid points needed to properly include `stop`
    num = int((stop - start) / step) + 2

    return start + np.arange(num) * step


def align_start_of_range(start, step, align):
    """
    Compute an aligned starting point of range

    The function returns a value `new_start` smaller or equal to `start`
    such that `align = start + N * step` for some integer `N`.

    :param start: Original starting point of range
    :param step: Range step size
    :param align: Value to use for alignment
    :return: An aligned starting point
    """
    offset = (start - align) % step
    return start - offset


def sparse_histogram_chunks_from_dataset_iterator(
        dset_in_iterator: typing.Iterable[xr.Dataset],
        bin_cols: list[str],
        weight_col: str,
) -> typing.Iterator[pd.DataFrame]:
    """
    We guarantee that the yielded data frames satisfy the following:
    
    1. They contain the columns ``bin_cols + [weight_col]``

    2. If concatenated, the data frame is sorted by bin_cols

    3. If concatenated, ``bin_cols`` have unique rows
    """
    out_chunks = []  # type: list[pd.DataFrame]

    for chunk_in in dset_in_iterator:
        # Group by bin idx and compute weight sum
        df = chunk_in[bin_cols + [weight_col]].to_dataframe()
        out_chunk = df.groupby(bin_cols).sum().reset_index()
        included = np.all(out_chunk[bin_cols].to_numpy() >= 0, axis=1)  # Remove out-of-bounds entries
        out_chunks.append(out_chunk.iloc[included])
    
    # Aggregate output chunks
    df_out = pd.concat(out_chunks, ignore_index=True)
    df_out = df_out.groupby(bin_cols).sum().reset_index()

    yield df_out
    

    #from pathlib import Path
    #import datetime
    #out_chunk_cols = list(bins) + ['weights']
    #out_chunk_bufsize = 10_000_000
    #num_output_df_written = 0
    #if dset_out.diskless():
    #    import tempfile
    #    writable_location = Path(tempfile.gettempdir())
    #else:
    #    writable_location = Path(dset_out.main_dataset.filepath()).parent
    #out_chunk_fname_pattern = str(
    #    writable_location / 
    #    f'crecontemp_{datetime.datetime.now():%Y%m%d%H%M%S}' / 
    #    'aggchunk.*.parquet'
    #)


    #        ## Persist to disk if necessary
#        #num_rows_total = sum(len(df) for df in out_chunks)
#        #if num_rows_total > out_chunk_bufsize:
#        #    # Aggregate output chunks
#        #    df_out = pd.concat(out_chunks, ignore_index=True)
#        #    df_out = df_out.groupby(list(bins)).sum().reset_index()
#
#            # Save to disk
#            fname = out_chunk_fname_pattern.replace('*', '{:05d}').format(num_output_df_written)
#            Path(fname).parent.mkdir(exist_ok=True)
#            df_out.to_parquet(fname, compression='snappy')
#            out_chunks = []
#            num_output_df_written += 1
#
#    # If there are any persisted chunks, do the same to the last one, for completeness
#    if num_output_df_written:
#        # Aggregate output chunks
#        df_out = pd.concat(out_chunks, ignore_index=True)
#        df_out = df_out.groupby(list(bins)).sum().reset_index()
#
#        # Save to disk
#        fname = out_chunk_fname_pattern.replace('*', '{:05d}').format(num_output_df_written)
#        Path(fname).parent.mkdir(exist_ok=True)
#        df_out.to_parquet(fname, compression='snappy')
#        out_chunks = []
#        num_output_df_written += 1
