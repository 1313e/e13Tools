# -*- coding: utf-8 -*-

"""
e13Tools
========
Provides a collection of functions that were created by `1313e`.
Recommended usage::

    import e13tools as e13

Available modules
-----------------
pyplot
    Provides a collection of functions useful in various plotting routines.
sampling
    Provides a collection of functions and techniques useful in sampling
    problems.

"""

import sys
import six
import distutils.version

from . import pyplot
from . import sampling
from .version import __version__


# List of version requirements
__version__numpy__ = str('1.6')
__version__mpl__ = str('1.4.3')


# Function to compare versions
def e13_compare_versions(a, b):
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

# Check for Python 3.3 or higher
if not sys.version_info[:2] >= (3, 3):
    raise ImportError("e13Tools requires Python 3.3 or later")

# Check for NumPy version
try:
    import numpy
except ImportError:
    raise ImportError("e13Tools requires NumPy")
else:
    if not e13_compare_versions(numpy.__version__, __version__numpy__):
        raise ImportError("NumPy %s was detected. e13Tools requires NumPy %s "
                          "or later" % (numpy.__version__, __version__numpy__))

# Check for Matplotlib version
try:
    import matplotlib
except ImportError:
    raise ImportError("e13Tools requires Matplotlib")
else:
    if not e13_compare_versions(matplotlib.__version__, __version__mpl__):
        raise ImportError("Matplotlib %s was detected. e13Tools requires "
                          "Matplotlib %s or later" % (matplotlib.__version__,
                                                      __version__mpl__))
