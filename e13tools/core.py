# -*- coding: utf-8 -*-

"""
Core
====
Provides a collection of functions that are core to **e13Tools** and are
imported automatically.

Available classes
-----------------
:class:`~InputError`
    Generic exception raised for errors in the function input arguments.

:class:`~ShapeError`
    Inappropriate argument shape (of correct type).

"""


# %% IMPORTS
from __future__ import absolute_import, division, print_function

import six
import distutils

__all__ = ['InputError', 'ShapeError']


# %% CLASSES
# Define Error class for wrong inputs
class InputError(Exception):
    """
    Generic exception raised for errors in the function input arguments.

    General purpose exception class, raised whenever the function input
    arguments prevent the correct execution of the function without specifying
    the type of error (eg. ValueError, TypeError, etc).

    """

    pass


# Define Error class for wrong shapes
class ShapeError(Exception):
    """
    Inappropriate argument shape (of correct type).

    """

    pass


# %% FUNCTIONS
def _compare_versions(a, b):
    "Return True if `a` is greater than or equal to `b`."
    if a:
        if six.PY3:
            if isinstance(a, bytes):
                a = a.decode('ascii')
            if isinstance(b, bytes):
                b = b.decode('ascii')
        a = distutils.version.LooseVersion(a)
        b = distutils.version.LooseVersion(b)
        return(a >= b)
    else:
        return(False)
