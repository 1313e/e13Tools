# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
import logging
from os import path
from sys import platform

# Package imports
from e13tools.core import InputError
import numpy as np
import pytest

# PRISM imports
from e13tools.utils import (docstring_append, docstring_copy,
                            docstring_substitute, check_instance,
                            convert_str_seq, delist, import_cmaps, raise_error,
                            raise_warning, rprint)

# Save the path to this directory
dirpath = path.dirname(__file__)

# Save if this platform is Windows
win32 = platform.startswith('win')


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for the custom function decorators
class TestDecorators(object):
    # Create method with no docstring that is appended
    @docstring_append("appended")
    def append_method1(self):
        pass

    # Create method with a docstring that is appended
    @docstring_append("appended")
    def append_method2(self):
        """original """

    # Check if docstring_append works correctly
    def test_docstring_append(self):
        assert self.append_method1.__doc__ == "appended"
        assert self.append_method2.__doc__ == "original appended"

    # Create method with no docstring at all
    def empty_method(self):
        pass

    # Create method that copies an empty docstring
    @docstring_copy(empty_method)
    def copy_method1(self):
        pass

    # Create method that copies a docstring
    @docstring_copy(append_method1)
    def copy_method2(self):
        pass

    # Check if docstring_copy works correctly
    def test_docstring_copy(self):
        assert self.copy_method1.__doc__ is None
        assert self.copy_method1.__doc__ == self.empty_method.__doc__
        assert self.copy_method2.__doc__ == self.append_method1.__doc__

    # Check if providing both args and kwargs raises an error
    with pytest.raises(InputError):
        @docstring_substitute("positional", x="keyword")
        def substitute_method1(self):
            pass

    # Create method using args substitutes
    @docstring_substitute("positional")
    def substitute_method2(self):
        """%s"""

    # Create method using kwargs substitutes
    @docstring_substitute(x="keyword")
    def substitute_method3(self):
        """%(x)s"""

    # Check if providing args to a method with no docstring raises an error
    with pytest.raises(InputError):
        @docstring_substitute("positional")
        def substitute_method4(self):
            pass

    # Check if docstring_substitute works correctly
    def test_docstring_substitute(self):
        assert self.substitute_method2.__doc__ == "positional"
        assert self.substitute_method3.__doc__ == "keyword"


# Pytest for the check_instance function
def test_check_instance():
    # Check if providing a non-class raises an error
    with pytest.raises(InputError):
        check_instance(np.array(1), np.array)

    # Check if providing an incorrect instance raises an error
    with pytest.raises(TypeError):
        check_instance(list(), np.ndarray)

    # Check if providing a proper instance of a class gives 1
    assert check_instance(np.array(1), np.ndarray) == 1


# Pytest for the convert_str_seq function
def test_convert_str_seq():
    # Check if string sequence is converted correctly
    assert convert_str_seq('[[]]]1e1,\n8.,A<{7)\\B') == [10., 8.0, 'A', 7, 'B']


# Pytest for the delist function
def test_delist():
    # Check if providing not a list raises an error
    with pytest.raises(TypeError):
        delist(np.array([1]))

    # Check if provided list is delisted correctly
    assert delist([[], (), [np.array(1)], [7], 8]) == [[np.array(1)], [7], 8]


# Pytest for the import_cmaps function
def test_import_cmaps():
    # Check if providing a non-existing directory raises an error
    with pytest.raises(OSError):
        import_cmaps('./test')

    # Check if providing a custom directory with invalid cmaps raises an error
    with pytest.raises(InputError):
        import_cmaps(path.join(dirpath, 'data'))


# Pytest for the raise_error function
def test_raise_error():
    # Create a logger and check if an error can be properly raised and logged
    logger = logging.getLogger('TEST')
    with pytest.raises(ValueError):
        raise_error('ERROR', ValueError, logger)


# Pytest for the raise_warning function
def test_raise_warning():
    # Create a logger and check if a warning can be properly raised and logged
    logger = logging.getLogger('TEST')
    with pytest.warns(UserWarning):
        raise_warning('WARNING', UserWarning, logger)


# Pytest for the rprint function
def test_rprint():
    # Check if rprint works correctly
    rprint('Testing')
