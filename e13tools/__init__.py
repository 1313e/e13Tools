# -*- coding: utf-8 -*-

"""
e13Tools
========
Provides a collection of functions that were created by **1313e**.
Recommended usage::

    import e13tools as e13

Available modules
-----------------
:mod:`~core`
    Provides a collection of functions that are core to **e13Tools** and are
    imported automatically.

:mod:`~math`
    Provides a collection of functions useful in various mathematical
    calculations and data array manipulations.

:mod:`~pyplot`
    Provides a collection of functions useful in various plotting routines.

:mod:`~sampling`
    Provides a collection of functions and techniques useful in sampling
    problems.

"""


# %% IMPORTS AND DECLARATIONS
from __future__ import absolute_import, division, print_function

# Import package modules
from .__version__ import version as __version__
from . import core
from .core import *
from . import math
from . import pyplot
from . import sampling

__all__ = ['math', 'pyplot', 'sampling']
__all__.extend(core.__all__)
