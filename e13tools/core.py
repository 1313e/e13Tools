# -*- coding: utf-8 -*-

"""
Core
====
Provides a collection of functions that are core to e13Tools and are imported
automatically.

"""


# %% IMPORTS
from __future__ import division, absolute_import, print_function

__all__ = ['InputError', 'ShapeError']


# %% FUNCTIONS
# Define Error class for wrong inputs
class InputError(Exception):
    """
    Generic exception raised for errors in a single or multiple function input
    argument(s).

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
