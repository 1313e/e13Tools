# -*- coding: utf-8 -*-

"""
Sampling
========
Provides a collection of functions and techniques useful in sampling problems.
Recommended usage::

    import e13tools.sampling as e13spl

Available submodules
--------------------
:func:`~lhs`
    Provides a Latin Hypercube Sampling method.

"""


# %% IMPORTS
from __future__ import absolute_import, division, print_function

from . import lhs
from .lhs import *

__all__ = []
__all__.extend(lhs.__all__)
