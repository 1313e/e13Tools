# -*- coding: utf-8 -*-

"""
Utilities
=========
Provides several useful utility functions.

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Built-in imports
from inspect import currentframe, getouterframes, isclass, isfunction, ismethod
import logging
import logging.config
import os
from os import path
import warnings

# Package imports
from matplotlib.cm import register_cmap
from matplotlib.colors import LinearSegmentedColormap as LSC
import numpy as np
from six import string_types

# e13Tools imports
from e13tools import InputError
try:
    from mpi4py import MPI
except ImportError:
    import e13tools.dummyMPI as MPI

# All declaration
__all__ = ['aux_char_list', 'check_instance', 'convert_str_seq', 'delist',
           'docstring_append', 'docstring_copy', 'docstring_substitute',
           'get_outer_frame', 'import_cmaps', 'raise_error', 'raise_warning',
           'rprint']

# Determine MPI size and ranks
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()


# %% DECORATOR DEFINITIONS
# Define custom decorator for appending docstrings to a function's docstring
def docstring_append(addendum, join=''):
    """
    Custom decorator that allows a given string `addendum` to be appended to
    the docstring of the target function, separated by a given string `join`.

    """

    def do_append(target):
        if target.__doc__:
            target.__doc__ = join.join([target.__doc__, addendum])
        else:
            target.__doc__ = addendum
        return(target)
    return(do_append)


# Define custom decorator for copying docstrings from one function to another
def docstring_copy(source):
    """
    Custom decorator that allows the docstring of a function `source` to be
    copied to the target function.

    """

    def do_copy(target):
        if source.__doc__:
            target.__doc__ = source.__doc__
        return(target)
    return(do_copy)


# Define custom decorator for substituting strings into a function's docstring
def docstring_substitute(*args, **kwargs):
    """
    Custom decorator that allows either given positional arguments `args` or
    keyword arguments `kwargs` to be substituted into the docstring of the
    target function.

    """

    if len(args) and len(kwargs):
        raise InputError("Either only positional or keyword arguments are "
                         "allowed!")
    else:
        params = args or kwargs

    def do_substitution(target):
        if target.__doc__:
            target.__doc__ = target.__doc__ % (params)
        else:
            raise InputError("Target has no docstring available for "
                             "substitutions!")
        return(target)
    return(do_substitution)


# %% FUNCTION DEFINITIONS
# This function checks if a given instance was initialized properly
def check_instance(instance, cls):
    """
    Checks if provided `instance` has been initialized from a proper `cls`
    (sub)class. Raises a :class:`~TypeError` if `instance` is not an instance
    of `cls`.

    Parameters
    ----------
    instance : object
        Class instance that needs to be checked.
    cls : class
        The class which `instance` needs to be properly initialized from.

    Returns
    -------
    result : bool
        Bool indicating whether or not the provided `instance` was initialized
        from a proper `cls` (sub)class.

    """

    # Check if cls is a class
    if not isclass(cls):
        raise InputError("Input argument 'cls' must be a class!")

    # Check if instance was initialized from a cls (sub)class
    if not isinstance(instance, cls):
        raise TypeError("Input argument 'instance' must be an instance of the "
                        "%s.%s class!" % (cls.__module__, cls.__name__))

    # Retrieve a list of all cls attributes
    class_attrs = dir(cls)

    # Check if all cls attributes can be called in instance
    for attr in class_attrs:
        if not hasattr(instance, attr):
            return(0)
    else:
        return(1)


# Function for converting a string sequence to a sequence of elements
def convert_str_seq(seq):
    """
    Converts a provided sequence to a string, removes all auxiliary characters
    from it, splits it up into individual elements and converts all elements
    back to integers, floats and/or strings.

    The auxiliary characters are given by the :obj:`~aux_char_list` list. One
    can add, change and remove characters from the list if required.

    Parameters
    ----------
    seq : str or array_like
        The sequence that needs to be converted to individual elements.
        If array_like, `seq` is first converted to a string.

    Returns
    -------
    new_seq : list
        A list with all individual elements converted to integers, floats
        and/or strings.

    """

    # Convert sequence to a string
    seq = str(seq)

    # Remove all unwanted characters from the string
    for char in aux_char_list:
        seq = seq.replace(char, ' ')

    # Split sequence up into elements
    seq = seq.split()

    # Loop over all elements in seq
    for i, val in enumerate(seq):
        # Try to convert to int or float
        try:
            # If string contains an E or e, check if it is a float
            if 'e' in val.lower():
                seq[i] = float(val)
            # If string contains a dot, check if it is a float
            elif '.' in val:
                seq[i] = float(val)
            # If string contains no dot, E or e, check if it is an int
            else:
                seq[i] = int(val)
        # If it cannot be converted to int or float, save as string
        except ValueError:
            seq[i] = val

    # Return it
    return(seq)


# List of auxiliary characters to be used in convert_str_seq()
aux_char_list = ['(', ')', '[', ']', ',', "'", '"', '|', '/', '{', '}', '<',
                 '>', '´', '¨', '`', '\\', '?', '!', '%', ';', '=', '$', '~',
                 '#', '@', '^', '&', '*', '“', '’', '”', '‘']


# Function that returns a copy of a list with all empty lists/tuples removed
def delist(list_obj):
    """
    Returns a copy of `list_obj` with all empty lists and tuples removed.

    Parameters
    ----------
    list_obj : list
        A list object that requires its empty list/tuple elements to be
        removed.

    Returns
    -------
    delisted_copy : list
        Copy of `list_obj` with all empty lists/tuples removed.

    """

    # Check if list_obj is a list
    if(type(list_obj) != list):
        raise TypeError("Input argument 'list_obj' is not of type 'list'!")

    # Make a copy of itself
    delisted_copy = list(list_obj)

    # Remove all empty lists/tuples from this copy
    off_dex = len(delisted_copy)-1
    for i, element in enumerate(reversed(delisted_copy)):
        # Remove empty lists
        if(isinstance(element, list) and element == []):
            delisted_copy.pop(off_dex-i)
        # Remove empty tuples
        elif(isinstance(element, tuple) and element == ()):
            delisted_copy.pop(off_dex-i)

    # Return the copy
    return(delisted_copy)


# This function retrieves a specified outer frame of a function
def get_outer_frame(name):
    """
    Checks whether or not the calling function contains an outer frame called
    `name` and returns it if so. If this frame cannot be found, returns *None*
    instead.

    Parameters
    ----------
    name : str or function
        If str, the name of the function whose frame must be located in the
        outer frames of the calling function.
        If function, its frame must be located in the outer frames instead.

    Returns
    -------
    outer_frame : frame or None
        The requested outer frame if it was found, or *None* if it was not.

    """

    # If name is a function or method, obtain its name
    if isfunction(name) or ismethod(name):
        name = name.__name__

    # Else, if name is not a string, raise an error
    elif not isinstance(name, string_types):
        raise InputError("Input argument 'name' must be a callable function or"
                         " method, or be of type 'str'!")

    # Obtain the caller's frame
    caller_frame = currentframe().f_back

    # Loop over all outer frames and check if name is in there
    for frame_info in getouterframes(caller_frame):
        if(frame_info[3] == name):
            return(frame_info[0])
    else:
        return(None)


# Function to import all custom colormaps in a directory
def import_cmaps(cmap_dir):
    """
    Reads in custom colormaps from a provided directory `cmap_dir`, transforms
    them into :obj:`~matplotlib.colors.LinearSegmentedColormap` objects and
    registers them in the :mod:`~matplotlib.cm` module. Both the imported
    colormap and its reversed version will be registered.

    Parameters
    ----------
    cmap_dir : str
        Relative or absolute path to the directory that contains custom
        colormap files. A colormap file can be a NumPy binary file ('.npy' or
        '.npz') or any text file.

    Notes
    -----
    All colormap files in `cmap_dir` must have names starting with 'cm\\_'. The
    resulting colormaps will have the name of their file without the prefix and
    extension.

    """

    # Obtain path to directory with colormaps
    cmap_dir = path.abspath(cmap_dir)

    # Check if provided directory exists
    if not path.exists(cmap_dir):
        raise OSError("Input argument 'cmap_dir' is a non-existing path (%r)!"
                      % (cmap_dir))

    # Obtain the names of all files in cmap_dir
    filenames = next(os.walk(cmap_dir))[2]
    cm_files = []

    # Extract the files with defined colormaps
    for filename in filenames:
        if(filename[:3] == 'cm_'):
            cm_files.append(filename)
    cm_files.sort()

    # Read in all the defined colormaps, transform and register them
    for cm_file in cm_files:
        # Split basename and extension
        base_str, ext_str = path.splitext(cm_file)
        cm_name = base_str[3:]

        # Process colormap files
        try:
            # Obtain absolute path to colormap data file
            cm_file_path = path.join(cmap_dir, cm_file)

            # Read in colormap data
            if ext_str in ('.npy', '.npz'):
                # If file is a NumPy binary file
                colorlist = np.load(cm_file_path).tolist()
            else:
                # If file is anything else
                colorlist = np.genfromtxt(cm_file_path).tolist()

            # Transform colorlist into a Colormap
            cmap = LSC.from_list(cm_name, colorlist, N=len(colorlist))
            cmap_r = LSC.from_list(cm_name+'_r', list(reversed(colorlist)),
                                   N=len(colorlist))

            # Add cmap to matplotlib's cmap list
            register_cmap(cmap=cmap)
            register_cmap(cmap=cmap_r)
        except Exception as error:
            raise InputError("Provided colormap %r is invalid! (%s)"
                             % (cm_name, error))


# This function raises a given error after logging the error
def raise_error(err_msg, err_type=Exception, logger=None):
    """
    Raises a given error `err_msg` of type `err_type` and logs the error using
    the provided `logger`.

    Parameters
    ----------
    err_msg : str
        The message included in the error.

    Optional
    --------
    err_type : :class:`Exception` subclass. Default: :class:`Exception`
        The type of error that needs to be raised.
    logger : :obj:`~logging.Logger` object or None. Default: None
        The logger to which the error message must be written.
        If *None*, the :obj:`~logging.RootLogger` logger is used instead.

    """

    # Log the error and raise it right after
    logger = logging.root if logger is None else logger
    logger.error(err_msg)
    raise err_type(err_msg)


# This function raises a given warning after logging the warning
def raise_warning(warn_msg, warn_type=UserWarning, logger=None, stacklevel=1):
    """
    Raises a given warning `warn_msg` of type `warn_type` and logs the warning
    using the provided `logger`.

    Parameters
    ----------
    warn_msg : str
        The message included in the warning.

    Optional
    --------
    warn_type : :class:`Warning` subclass. Default: :class:`UserWarning`
        The type of warning that needs to be raised.
    logger : :obj:`~logging.Logger` object or None. Default: None
        The logger to which the warning message must be written.
        If *None*, the :obj:`~logging.RootLogger` logger is used instead.
    stacklevel : int. Default: 1
        The stack level of the warning message at the location of this function
        call. The actual used stack level is increased by one.

    """

    # Log the warning and raise it right after
    logger = logging.root if logger is None else logger
    logger.warning(warn_msg)
    warnings.warn(warn_msg, warn_type, stacklevel=stacklevel+1)


# Redefine the print function to include the MPI rank if MPI is used
def rprint(*args, **kwargs):
    """
    Custom :func:`~print` function that prepends the rank of the MPI process
    that calls it to the message if the size of the intra-communicator is more
    than 1.
    Takes the same input arguments as the normal :func:`~print` function.

    """

    # If MPI is used and size > 1, prepend rank to message
    if(MPI.__name__ == 'mpi4py.MPI' and size > 1):
        args = list(args)
        args.insert(0, "Rank %i:" % (rank))
    print(*args, **kwargs)
