# -*- coding: utf-8 -*-

"""
PyPlot
======
Provides a collection of functions useful in various plotting routines.
Recommended usage::

    import e13tools.pyplot as e13plt

Available submodules
--------------------
core
    Provides a collection of functions that are core to PyPlot and are imported
    automatically.

"""


# %% IMPORTS
from __future__ import division, absolute_import, print_function

from . import core
from .core import *

__all__ = []
__all__.extend(core.__all__)


# %% DOCTEST
if __name__ == '__main__':
    import doctest
    doctest.testmod()
