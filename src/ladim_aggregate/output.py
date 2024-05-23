import netCDF4 as nc
import numpy as np
import logging


logger = logging.getLogger(__name__)


class MultiDataset:
    """Encapsulates a collection of netCDF4 datasets.

    The class contains a main dataset which holds all the coordinate variables.
    Some of the coordinate variables are defined as "cross-dataset"
    coordinates. Variables having any of the cross-dataset dimensions are
    spread across multiple datasets. Variables with only "regular" coordinates
    are kept in the main dataset.
    """

    def __init__(self, filename, **kwargs):
        """
        Initialize a MultiDataset. The file of the main dataset is given
        explicitly. The names of the subdatasets are constructed from the main
        filename, of the form "maindatasetname_firstcoordvalue_secondcoordvalue".

        :param filename: Name of the main dataset.
        :param kwargs: Additional arguments passed to netCDF4.Dataset
        """
        self._dataset_kwargs = kwargs
        self.datasets = dict()
        self._cross_coords = dict()
        self._cross_vars = dict()
        self._editable = True  # When subdatasets are created, no more variables can be added

        if isinstance(filename, nc.Dataset):
            self.main_dataset = filename
            self._has_diskless_input_dataset = True
            self._dataset_kwargs['diskless'] = True
        else:
            self.main_dataset = nc.Dataset(filename, mode='w', **kwargs)
            self._has_diskless_input_dataset = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        logger.info(f'Close dataset "{self.main_dataset.filepath()}"')
        if not self._has_diskless_input_dataset:
            self.main_dataset.close()
        for d in self.datasets.values():
            logger.info(f'Close dataset "{d.filepath()}"')
            d.close()

    def createCoord(self, varname, data, attrs=None, cross_dataset=False):
        """
        Create coordinate variable and dimension in the main dataset.

        Coordinate variables cannot be created after setData or getData is used
        on a cross-dataset variable.

        :param varname: Name of coordinate variable and dimension
        :param data: Data used to initialize variable
        :param attrs: Optional coordinate attributes
        :param cross_dataset: True if this is a cross-dataset dimension
        :return: The newly created variable
        """
        if not self._editable:
            raise TypeError(
                "Adding new variables must be done before accessing cross-dataset data")

        data = np.array(data)

        if np.issubdtype(data.dtype, np.datetime64):
            attrs = {
                **(attrs or dict()),
                **dict(
                    units="microseconds since 1970-01-01",
                    calendar="proleptic_gregorian",
                ),
            }
            datediff = data - np.datetime64('1970-01-01')
            data = datediff.astype('timedelta64[us]').astype('i8')

        dset = self.main_dataset
        dset.createDimension(varname, len(data))
        variable = dset.createVariable(varname, data.dtype, varname, fill_value=False)
        variable.set_auto_maskandscale(False)
        variable[:] = data
        if attrs:
            dset.variables[varname].setncatts(attrs)
        if cross_dataset:
            self._cross_coords[varname] = data
        return variable

    def setAttrs(self, varname, attrs):
        """
        Set attributes of a regular variable
        :param varname:
        :param attrs:
        :return:
        """
        if not self._editable:
            # We cannot edit attributes of any variables. If we were to allow this, we
            # would need to change attributes of all sub-datasets as well.
            raise TypeError(
                "Setting attributes must be done before accessing cross-dataset data")

        # Set cross-dataset variable attributes
        if varname in self._cross_vars:
            old_attrs = self._cross_vars[varname]['attrs']
            self._cross_vars[varname]['attrs'] = {**old_attrs, **attrs}

        # Set regular variable attributes
        else:
            v = self.main_dataset.variables[varname]
            v = v  # type: nc.Variable
            v.setncatts(attrs)

    def getCoord(self, varname):
        """
        Return a coordinate variable from the main dataset
        :param varname: Variable name
        :return: The variable object
        """
        return self.main_dataset.variables[varname]

    def getAttrs(self, varname):
        """
        Return the attributes of either a regular or cross-dataset
        variable.

        :param varname: Variable name
        :return: A dict of attributes
        """

        # Get cross-dataset variable attributes
        if varname in self._cross_vars:
            return self._cross_vars[varname]['attrs']

        # Get regular variable attributes
        else:
            v = self.main_dataset.variables[varname]
            return {k: v.getncattr(k) for k in v.ncattrs()}

    def createVariable(self, varname, data, dims, attrs=None):
        """
        Create a dataset variable. If any of the dimensions are cross-dataset,
        the variable will be spread out over multiple datasets.

        Variables cannot be created after setData or getData is used on a
        cross-dataset variable.

        :param varname: Variable name
        :param data: The initial data of the variable
        :param dims: A tuple of dimension names
        :param attrs: An optional dict of attributes
        :return: The variable object if regular variable, None otherwise
        """
        if not self._editable:
            raise TypeError(
                "Adding new variables must be done before accessing cross-dataset data")

        data = np.array(data)

        # A single dimension should be converted to list
        if isinstance(dims, str):
            dims = [dims]

        # Create cross-dataset variable
        if any(d in self._cross_coords for d in dims):
            self._cross_vars[varname] = dict(
                data=data,
                dims=dims,
                attrs=attrs or dict(),
            )
            variable = None

        # Create regular variable
        else:
            dset = self.main_dataset
            variable = dset.createVariable(varname, data.dtype, dims, fill_value=False)
            variable.set_auto_maskandscale(False)
            variable[:] = data
            if attrs:
                variable.setncatts(attrs)

        return variable

    def _get_filename(self, file_idx):
        import os

        # Get filename-friendly value of variable
        def strval(varname, index):
            variable = self.getCoord(varname)
            attrs = self.getAttrs(varname)
            value = variable[index]
            if 'since' in attrs.get('units', ''):
                import cftime
                date = cftime.num2date(value, attrs['units'],
                                       attrs.get('calendar', 'standard'))
                return f'{date:%Y%m%d%H%M%S}'
            else:
                return str(value)

        stubs = [str(strval(k, v)) for k, v in file_idx.items()]
        base, ext = os.path.splitext(self.main_dataset.filepath())
        return base + '_' + '_'.join(stubs) + ext

    def getDataset(self, **file_idx):
        """Return sub-dataset, or create it if necessary"""
        file_idx_key = tuple((k, file_idx[k]) for k in self._cross_coords.keys())
        if file_idx_key not in self.datasets:
            fname = self._get_filename(file_idx)
            logger.info(f'Create sub-dataset "{fname}"')
            dset = nc.Dataset(fname, mode='w', **self._dataset_kwargs)
            self._copy_from_main(dset, file_idx)
            self.datasets[file_idx_key] = dset
            self._editable = False

        return self.datasets[file_idx_key]

    def _copy_from_main(self, dest, crossdim_idx):
        cross_dims = list(self._cross_coords.keys())
        source = self.main_dataset

        # Copy dimensions
        for dim in source.dimensions.values():
            if dim.name in cross_dims:
                dest.createDimension(dim.name, 1)
            else:
                dest.createDimension(dim.name, dim.size)

        # Copy variables
        for v in source.variables.values():
            new_var = dest.createVariable(v.name, v.dtype, v.dimensions)
            new_var.set_auto_maskandscale(False)
            atts = {k: v.getncattr(k) for k in v.ncattrs()}
            new_var.setncatts(atts)
            if v.name in cross_dims:
                # If crossdim coordinate, copy only one value
                global_idx = crossdim_idx[v.name]
                new_var[:] = v[global_idx]
                new_var.setncattr('global_index', global_idx)
            else:
                # Otherwise, copy all data
                new_var[:] = v[:]

        # Add filesplit variables
        for vname, vinfo in self._cross_vars.items():
            new_var = dest.createVariable(vname, vinfo['data'].dtype, vinfo['dims'])
            new_var.set_auto_maskandscale(False)
            new_var[:] = vinfo['data']
            dest.variables[vname].setncatts(vinfo['attrs'])

        # Copy global attributes
        atts = {k: source.getncattr(k) for k in source.ncattrs()}
        dest.setncatts(atts)

    def _get_shape_of_variable_data(self, varname, idx):
        shape = []
        for dim_slice, dim_name in zip(idx, self._cross_vars[varname]['dims']):
            dim_size = self.main_dataset.dimensions[dim_name].size
            dim_range = range(*dim_slice.indices(dim_size))
            shape.append(len(dim_range))
        return tuple(shape)

    def getData(self, varname, idx=None):
        """
        Get data from a dataset variable.

        :param varname: Variable name
        :param idx: A tuple of slices, or None if the entire data range is desired.
        :return: A numpy array of data
        """

        if varname in self._cross_vars:
            variable_info = self._cross_vars[varname]
            if idx is None:
                idx = (slice(None),) * len(variable_info['dims'])

            shape = self._get_shape_of_variable_data(varname, idx)
            data = np.empty(shape, variable_info['data'].dtype)

            for i_file, i_var, i_data in self._split_indices(varname, idx):
                dset = self.getDataset(**i_file)
                data[i_data] = dset.variables[varname][i_var]

            return data

        else:
            if idx is None:
                idx = slice(None)
            return self.main_dataset.variables[varname][idx]

    def setData(self, varname, data, idx=None):
        """
        Set data of a dataset variable

        :param varname: Variable name
        :param data: A numpy array of data
        :param idx: A tuple of slices, or None if the entire data range is desired.
        :return: The input data
        """
        if varname in self._cross_vars:
            variable_info = self._cross_vars[varname]
            if idx is None:
                idx = (slice(None),) * len(variable_info['dims'])

            shape = self._get_shape_of_variable_data(varname, idx)
            bdata = np.broadcast_to(data, shape)

            for i_file, i_var, i_data in self._split_indices(varname, idx):
                dset = self.getDataset(**i_file)

                dset.variables[varname][i_var] = bdata[i_data]

            return data

        else:
            if idx is None:
                idx = slice(None)
            self.main_dataset.variables[varname][idx] = data
        return data

    def incrementData(self, varname, data, idx=None):
        """
        Increment a dataset variable

        :param varname: Variable name
        :param data: A numpy array of data
        :param idx: A tuple of slices, or None if the entire data range is desired.
        :return: The input data
        """
        previous = self.getData(varname, idx)
        newdata = previous + np.broadcast_to(data, previous.shape)
        self.setData(varname, newdata, idx)

    def _split_indices(self, varname, index):
        """
        Iterate over subdatasets to aid access of cross-dataset variables.

        The input index is a tuple of slices, relative to the global shape
        of the variable. The function iterates over each dataset spanned by
        the range.

        For each iteration, returns a tuple `(cross_idx, var_idx, data_idx)`
        representing the portion of the data spanned by a single dataset.
        - `cross_idx` is a dict of the form `{crossdimA: intA, crossdimB: intB, ...}`,
          identifying the subdataset.
        - `var_idx` is a tuple of either zeros or slices. For each
          cross-dataset dimension there is a zero, and for each main dimension
          there is a slice (taken directly from `index`). This tuple can be
          used to directly index variables in a subdataset.
        - `data_idx` is a tuple of slice(None) and integers. For each cross-
          dataset dimension there is an integer, and for each main dimension
          there is slice(None). The integer is relative to the shape of the
          data subset.
        :param varname: The variable name
        :param index: A tuple of slices
        :return: A tuple `(cross_idx, var_idx, data_idx)` representing a
        portion of the data spanned by a single subdataset.
        """

        from itertools import product

        varinfo = self._cross_vars[varname]
        dims = varinfo['dims']
        is_splitdim = [dim in self._cross_coords for dim in dims]
        big_shape = [self.main_dataset.dimensions[d].size for d in dims]
        big_range = [range(*i.indices(s)) for i, s in zip(index, big_shape)]
        shape = [len(range(*i.indices(s))) for i, s in zip(index, big_shape)]

        # Construct the var_idx, which is the index of the subdataset variable
        var_idx = []
        for dim, index_element, sd in zip(dims, index, is_splitdim):
            if sd:
                var_idx.append(0)
            else:
                var_idx.append(index_element)

        # Construct the index elements relative to the big data array
        data_indices = []
        for index_element, dim, sz, sd in zip(index, dims, shape, is_splitdim):
            if sd:
                data_indices.append(range(sz))
            else:
                data_indices.append([slice(None)])

        # Iterate over the cartesian product
        for data_idx in product(*data_indices):
            file_idx = {d: br[di] for d, br, di, sd in
                        zip(dims, big_range, data_idx, is_splitdim) if sd}
            yield file_idx, tuple(var_idx), data_idx

    def to_dict(self):
        all_datasets = [self.main_dataset] + list(self.datasets.values())
        dset_dict = {}
        for dset in all_datasets:
            key = dset.filepath()
            dset_dict[key] = nc_to_dict(dset)
        return dset_dict


def nc_to_dict(dset):
    d = dict()
    for varname in dset.variables:
        v = dset.variables[varname]
        d[varname] = dict(
            dims=list(v.dimensions),
            data=v[:].tolist(),
        )
        if len(d[varname]['dims']) == 1:
            d[varname]['dims'] = d[varname]['dims'][0]

        atts = dict()
        for attname in v.ncattrs():
            atts[attname] = v.getncattr(attname)
        if atts:
            d[varname]['attrs'] = atts
    return d
