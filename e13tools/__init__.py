# -*- coding: utf-8 -*-

"""
e13Tools
========
Provides a collection of utility functions that were created by **1313e**.
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

:mod:`~utils`
    Provides several useful utility functions.

"""


# %% IMPORTS AND DECLARATIONS
from __future__ import absolute_import, division, print_function

# Import package modules
from .__version__ import __version__
from . import core
from .core import *
from . import math
from .math import *
from . import pyplot
from .pyplot import *
from . import sampling
from .sampling import *
from . import utils
from .utils import *

__all__ = ['math', 'pyplot', 'sampling', 'utils']
__all__.extend(core.__all__)
__all__.extend(math.__all__)
__all__.extend(pyplot.__all__)
__all__.extend(sampling.__all__)
__all__.extend(utils.__all__)
