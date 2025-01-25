"""Utility classes and modules."""
import ast
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Dict


class LazyDict(Mapping):
    """
    This is a lazy-loading dictionary.

    It makes working with dictionaries which have expensive-to-compute
    values less painful. Taken from:
    https://stackoverflow.com/questions/16669367/setup-dictionary-lazily.
    """

    def __init__(self, *args, **kwargs):
        """
        Create a LazyDict.

        Behaves like a regular dict once created, but
        creating it is a bit different. Consider a dict:
        {
            "key_1": expensive_fn(input_1),
            "key_2": expensive_fn(input_2),
            ...
        }

        To create an equivalent LazyDict:
        LazyDict({
            "key_1": (expensive_fn, input_1),
            "key_2": (expensive_fn, input_2),
            ...
        })
        """
        self._raw_dict = dict(*args, **kwargs)

    def __getitem__(self, key):
        func, arg = self._raw_dict.__getitem__(key)
        return func(arg)

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)

def json_dict_loader(json_path: Path) -> Dict:
    """
    Load the json file at json_path, and return the data as a dict.
    """
    with json_path.open('r') as json_file:
        raw_data = json.load(json_file)
        # Safer to use ast.literal_eval than eval to convert data keys to tuples.
        # The latter can execute arbitrary potentially malicious code while
        # the worst case attack vector for literal_eval would be to crash
        # the python process.
        # See https://docs.python.org/3/library/ast.html#ast.literal_eval
        # for more information.
        result_dict = {ast.literal_eval(key): value for key, value in raw_data.items()}
    return result_dict
