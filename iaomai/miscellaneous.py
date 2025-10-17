import os
import pandas as pd
from pathlib import Path
from functools import wraps


def simple_cache_csv_dataframe_fetch(target_file: Path):
    """
    Decorator to cache output of a function
    Checks if `target_file` exists
    - If not, executes the function and dumps the output to csv
    - If yes, reads the contents of `target_file`
    """

    def cache_decorator(func: callable) -> callable:
        @wraps(func)
        def new_func(*args, **kwargs):
            if os.path.isfile(target_file):
                data_to_dump = pd.read_csv(target_file, index_col=0)
                data_to_dump.index = pd.to_datetime(data_to_dump.index, utc=True)
            else:
                data_to_dump = func(*args, **kwargs)
                data_to_dump.to_csv(target_file)
            return data_to_dump

        return new_func

    return cache_decorator
