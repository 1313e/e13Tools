# -*- coding: utf-8 -*-

# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Built-in imports
import logging
from os import path
from sys import platform
from inspect import currentframe

# Package imports
import numpy as np
import pytest

# e13Tools imports
from e13tools.core import InputError
from e13tools.utils import (docstring_append, docstring_copy,
                            docstring_substitute, check_instance,
                            convert_str_seq, delist, get_outer_frame,
                            raise_error, raise_warning)

# Save the path to this directory
dirpath = path.dirname(__file__)

# Save if this platform is Windows
win32 = platform.startswith('win')


# %% CUSTOM CLASSES
# Define test class for get_outer_frame function testing
class _Test(object):
    def __init__(self):
        self._test()

    def _test(self):
        _test2(self)


def _test2(instance):
    get_outer_frame(instance.__init__)


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

    # Create old-style class with no docstring that is appended
    @docstring_append("appended")
    class append_old_class1:
        pass

    # Create old-style class with a docstring that is appended
    @docstring_append("appended")
    class append_old_class2:
        """original """

    # Create new-style class with no docstring that is appended
    @docstring_append("appended")
    class append_new_class1(object):
        pass

    # Create new-style class with a docstring that is appended
    @docstring_append("appended")
    class append_new_class2(object):
        """original """

    # Check if docstring_append works correctly
    def test_docstring_append(self):
        assert self.append_method1.__doc__ == "appended"
        assert self.append_method2.__doc__ == "original appended"
        assert self.append_old_class1.__doc__ == "appended"
        assert self.append_old_class2.__doc__ == "original appended"
        assert self.append_new_class1.__doc__ == "appended"
        assert self.append_new_class2.__doc__ == "original appended"
        assert self.append_new_class1.__name__ == 'append_new_class1'
        assert self.append_new_class1.__module__ != 'e13tools.utils'
        assert self.append_new_class2.__name__ == 'append_new_class2'
        assert self.append_new_class2.__module__ != 'e13tools.utils'

    # Create method with no docstring at all
    def empty_method(self):
        pass

    # Create new-style class with no docstring at all
    class empty_class(object):
        pass

    # Create method that copies an empty docstring
    @docstring_copy(empty_method)
    def copy_method1(self):
        pass

    # Create method that copies a docstring
    @docstring_copy(append_method2)
    def copy_method2(self):
        pass

    # Create old-style class that copies an empty docstring
    @docstring_copy(empty_class)
    class copy_old_class1:
        pass

    # Create old-style class that copies a docstring
    @docstring_copy(append_old_class2)
    class copy_old_class2:
        pass

    # Create new-style class that copies an empty docstring
    @docstring_copy(empty_class)
    class copy_new_class1(object):
        pass

    # Create new-style class that copies a docstring
    @docstring_copy(append_new_class2)
    class copy_new_class2(object):
        pass

    # Check if docstring_copy works correctly
    def test_docstring_copy(self):
        assert self.copy_method1.__doc__ is None
        assert self.copy_method1.__doc__ == self.empty_method.__doc__
        assert self.copy_method2.__doc__ == self.append_method2.__doc__
        assert self.copy_old_class1.__doc__ is None
        assert self.copy_old_class1.__doc__ == self.empty_class.__doc__
        assert self.copy_old_class2.__doc__ == self.append_old_class2.__doc__
        assert self.copy_new_class1.__doc__ is None
        assert self.copy_new_class1.__doc__ == self.empty_class.__doc__
        assert self.copy_new_class2.__doc__ == self.append_new_class2.__doc__
        assert self.copy_new_class1.__name__ == 'copy_new_class1'
        assert self.copy_new_class1.__module__ != 'e13tools.utils'
        assert self.copy_new_class2.__name__ == 'copy_new_class2'
        assert self.copy_new_class2.__module__ != 'e13tools.utils'

    # Check if providing both args and kwargs raises an error, method
    with pytest.raises(InputError):
        @docstring_substitute("positional", x="keyword")
        def substitute_method1(self):
            pass

    # Check if providing both args and kwargs raises an error, old-style class
    with pytest.raises(InputError):
        @docstring_substitute("positional", x="keyword")
        class substitute_old_class1:
            pass

    # Check if providing both args and kwargs raises an error, new-style class
    with pytest.raises(InputError):
        @docstring_substitute("positional", x="keyword")
        class substitute_new_class1(object):
            pass

    # Create method using args substitutes
    @docstring_substitute("positional")
    def substitute_method2(self):
        """%s"""

    # Create method using kwargs substitutes
    @docstring_substitute(x="keyword")
    def substitute_method3(self):
        """%(x)s"""

    # Create old-style class using args substitutes
    @docstring_substitute("positional")
    class substitute_old_class2:
        """%s"""

    # Create old-style class using kwargs substitutes
    @docstring_substitute(x="keyword")
    class substitute_old_class3:
        """%(x)s"""

    # Create new-style class using args substitutes
    @docstring_substitute("positional")
    class substitute_new_class2(object):
        """%s"""

    # Create new-style class using kwargs substitutes
    @docstring_substitute(x="keyword")
    class substitute_new_class3(object):
        """%(x)s"""

    # Check if providing args to a method with no docstring raises an error
    with pytest.raises(InputError):
        @docstring_substitute("positional")
        def substitute_method4(self):
            pass

    # Check providing args to an old_style class with no docstring
    with pytest.raises(InputError):
        @docstring_substitute("positional")
        class substitute_old_class4:
            pass

    # Check providing args to a new_style class with no docstring
    with pytest.raises(InputError):
        @docstring_substitute("positional")
        class substitute_new_class4(object):
            pass

    # Check if docstring_substitute works correctly
    def test_docstring_substitute(self):
        assert self.substitute_method2.__doc__ == "positional"
        assert self.substitute_method3.__doc__ == "keyword"
        assert self.substitute_old_class2.__doc__ == "positional"
        assert self.substitute_old_class3.__doc__ == "keyword"
        assert self.substitute_new_class2.__doc__ == "positional"
        assert self.substitute_new_class3.__doc__ == "keyword"
        assert self.substitute_new_class2.__name__ == 'substitute_new_class2'
        assert self.substitute_new_class2.__module__ != 'e13tools.utils'
        assert self.substitute_new_class3.__name__ == 'substitute_new_class3'
        assert self.substitute_new_class3.__module__ != 'e13tools.utils'


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
    assert convert_str_seq('[[]1e1,\n8.,A<{7)\\\\') == [10., 8.0, 'A', 7, '\\']


# Pytest for the delist function
def test_delist():
    # Check if providing not a list raises an error
    with pytest.raises(TypeError):
        delist(np.array([1]))

    # Check if provided list is delisted correctly
    assert delist([[], (), [np.array(1)], [7], 8]) == [[np.array(1)], [7], 8]


# Pytest for the get_outer_frame function
def test_get_outer_frame():
    # Check if providing a wrong argument raises an error
    with pytest.raises(InputError):
        get_outer_frame('test')

    # Check if providing a non-valid frame function returns None
    assert get_outer_frame(get_outer_frame) is None

    # Check if providing a valid function returns that frame
    caller_frame = currentframe()
    assert get_outer_frame(test_get_outer_frame) == caller_frame

    # Check if providing a valid method returns the correct method
    _Test()


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
