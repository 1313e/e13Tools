# -*- coding: utf-8 -*-

"""
e13Tools
========
Provides a collection of functions that were created by `1313e`.
Recommended usage::

    import e13tools as e13

Available modules
-----------------
core
    Provides a collection of functions that are core to e13Tools and are
    imported automatically.
math
    Provides a collection of functions useful in various mathematical
    calculations.
pyplot
    Provides a collection of functions useful in various plotting routines.
sampling
    Provides a collection of functions and techniques useful in sampling
    problems.

"""


# %% IMPORTS AND DECLARATIONS
from __future__ import division, absolute_import, print_function

import sys
import six
import distutils.version
from os import path

# Import package modules
from . import core
from .core import *
from . import math
from . import pyplot
from . import sampling

__all__ = ['math', 'pyplot', 'sampling']
__all__.extend(core.__all__)


# Get the version
_root_dir = path.abspath(path.dirname(__file__))
with open(path.join(_root_dir, 'VERSION'), 'r',
          encoding='utf-8') as _version_file:
    __version__ = _version_file.read().strip()

# List of version requirements
__version__numpy__ = str('1.8')
__version__mpl__ = str('1.4.3')
__version__astropy__ = str('1.3')


# %% FUNCTIONS AND CHECKS
# Function to compare versions
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

# Check for Python 2.7 or 3.4 and higher
if sys.version_info[0] == 2:
    if not sys.version_info[1] == 7:
        raise ImportError("e13Tools requires Python 2.7")
elif not sys.version_info[:2] >= (3, 4):
    raise ImportError("e13Tools requires Python 3.4 or later")

# Check for NumPy version
try:
    import numpy
except ImportError:
    raise ImportError("e13Tools requires NumPy")
else:
    if not _compare_versions(numpy.__version__, __version__numpy__):
        raise ImportError("NumPy %s was detected. e13Tools requires NumPy %s "
                          "or later" % (numpy.__version__, __version__numpy__))

# Check for Matplotlib version
try:
    import matplotlib
except ImportError:
    raise ImportError("e13Tools requires Matplotlib")
else:
    if not _compare_versions(matplotlib.__version__, __version__mpl__):
        raise ImportError("Matplotlib %s was detected. e13Tools requires "
                          "Matplotlib %s or later" % (matplotlib.__version__,
                                                      __version__mpl__))

try:
    import astropy
except ImportError:
    raise ImportError("e13Tools requires AstroPy")
else:
    if not _compare_versions(astropy.__version__, __version__astropy__):
        raise ImportError("AstroPy %s was detected. e13Tools requires "
                          "AstroPy %s or later" % (astropy.__version__,
                                                   __version__astropy__))
