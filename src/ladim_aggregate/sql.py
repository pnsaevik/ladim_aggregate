import sqlite3
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np


class Particles:
    """SQL-based backend for aggregation operations on particles"""
    def __init__(self, filename=None):
        self.filename = filename
        self.connection = None
        self.cursor = None
        self._tables = {}

    def __enter__(self):
        # Return error if file exists. The backend should be a temporary file.
        if self.filename and Path(self.filename).exists():
            raise FileExistsError

        self.connection = sqlite3.connect(self.filename or ':memory:')
        self.cursor = self.connection.cursor()

        self._add_col(tab='particle_instance', col='pid', dtype='INTEGER')
        self._add_col(tab='particle_instance', col='time_idx', dtype='INTEGER')
        self._add_col(tab='time', col='time_idx', dtype='INTEGER')
        self._add_col(tab='time', col='time', dtype='TEXT')
        self._add_col(tab='time', col='particle_count', dtype='INTEGER')
        self._add_col(tab='particle', col='pid', dtype='INTEGER')

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            # Do not commit changes on exit. Instead, assume that the user
            # has already extracted all the information they need.
            self.connection.close()
            self.connection = None

        if self.filename:
            Path(self.filename).unlink(missing_ok=True)

    def _add_col(self, tab, col, dtype):
        if tab not in self._tables:
            self._tables[tab] = pd.DataFrame(data={col: [dtype]})
            self.cursor.execute(f'CREATE TABLE {tab} ({col} {dtype})')

        if col not in self._tables[tab].columns:
            self._tables[tab][col] = dtype
            self.cursor.execute(f'ALTER TABLE {tab} ADD COLUMN {col} {dtype}')

    def _add_data(self, tab, cols, vals):
        self.cursor.executemany(
            f'INSERT INTO {tab} ({", ".join(cols)}) '
            f'VALUES ({", ".join(["?"] * len(cols))})',
            list(map(list, zip(*vals)))
        )

    def add_ladim(self, dset: xr.Dataset):
        # Add columns if necessary
        for varname in dset.variables:
            v = dset.variables[varname]
            dtype, data = get_sql_data_format(v.values)
            self._add_col(tab=v.dims[0], col=varname, dtype=dtype)

        # Add data
        for tab in ['time', 'particle_instance', 'particle']:
            cols = [c for c in self._tables[tab].columns if c in dset.variables]
            np_vals = (dset.variables[c].values for c in cols)
            vals = [get_sql_data_format(np_val)[1] for np_val in np_vals]
            self._add_data(tab, cols, vals)


def get_sql_data_format(np_data):
    out_data = np_data
    np_dtype = np.asarray(np_data).dtype
    if np.issubdtype(np_dtype, np.integer):
        sql_dtype = 'INTEGER'
    elif np.issubdtype(np_dtype, np.float):
        sql_dtype = 'REAL'
    elif np.issubdtype(np_dtype, np.str):
        sql_dtype = 'TEXT'
    elif np.issubdtype(np_dtype, np.datetime64):
        out_data = np_data.astype('datetime64[s]').astype(str)
        sql_dtype = 'TEXT'
    else:
        raise TypeError
    return sql_dtype, out_data
