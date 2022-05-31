import sqlite3
from pathlib import Path


class Particles:
    """SQL-based backend for aggregation operations on particles"""
    def __init__(self, filename=None):
        self.filename = filename
        self.connection = None

    def __enter__(self):
        # Return error if file exists. The backend should be a temporary file.
        if self.filename and Path(self.filename).exists():
            raise FileExistsError

        self.connection = sqlite3.connect(self.filename or ':memory:')
        self.cursor = self.connection.cursor()
        self.cursor.execute(
            'CREATE TABLE particle_instance('
            '  pid INTEGER'
            '  time_idx INTEGER'
            ')'
        )
        self.cursor.execute(
            'CREATE TABLE time('
            '  time_idx INTEGER'
            '  time TEXT'
            '  particle_count INTEGER'
            ')'
        )
        self.cursor.execute(
            'CREATE TABLE particle('
            '  pid INTEGER'
            ')'
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            # Do not commit changes on exit. Instead, assume that the user
            # has already extracted all the information they need.
            self.connection.close()
            self.connection = None

        if self.filename:
            Path(self.filename).unlink(missing_ok=True)
