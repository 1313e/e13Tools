# -*- coding: utf-8 -*-

"""
Math
====
Provides a collection of functions useful in various mathematical calculations.
Recommended usage::

    import e13tools.math as e13m

Available submodules
--------------------
core
    Provides a collection of functions that are core to Math and are imported
    automatically.

"""


# %% IMPORTS
from __future__ import division, absolute_import, print_function

from . import core
from .core import *

__all__ = []
__all__.extend(core.__all__)
