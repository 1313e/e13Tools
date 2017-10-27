# -*- coding: utf-8 -*-

"""
Sampling
======
Provides a collection of functions and techniques useful in sampling problems.
Recommended usage::

    import e13tools.sampling as e13spl

Available submodules
--------------------
lhcs
    Provides a Latin Hypercube Sampling method.

"""

from __future__ import division, absolute_import, print_function

from .lhcs import lhs

__all__ = ['lhs']
