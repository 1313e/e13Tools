# -*- coding: utf-8 -*-

# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Built-in imports
from os import path
from sys import platform

# Package imports
import pytest

# e13Tools imports
from e13tools.core import InputError
from e13tools.pyplot import import_cmaps

# Save the path to this directory
dirpath = path.dirname(__file__)

# Save if this platform is Windows
win32 = platform.startswith('win')


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for the import_cmaps function
def test_import_cmaps():
    # Check if providing a non-existing directory raises an error
    with pytest.raises(OSError):
        import_cmaps('./test')

    # Check if providing a custom directory with invalid cmaps raises an error
    with pytest.raises(InputError):
        import_cmaps(path.join(dirpath, 'data'))
