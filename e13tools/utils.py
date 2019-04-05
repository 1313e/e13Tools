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
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap as LSC
import numpy as np
from six import PY2

# e13Tools imports
from e13tools import InputError

# All declaration
__all__ = ['aux_char_set', 'check_instance', 'convert_str_seq', 'delist',
           'docstring_append', 'docstring_copy', 'docstring_substitute',
           'get_outer_frame', 'import_cmaps', 'raise_error', 'raise_warning']


# %% DECORATOR DEFINITIONS
# Define custom decorator for appending docstrings to a function's docstring
def docstring_append(addendum, join=''):
    """
    Custom decorator that allows a given string `addendum` to be appended to
    the docstring of the target function/class, separated by a given string
    `join`.

    Note
    ----
    In Python 2, classes that inherit the :class:`~object` class cannot have
    their docstrings modified after it has been defined. In this case, this
    decorator makes a subclass of the target class, sets the docstring during
    class definition and returns that instead. The returned subclass is
    functionally exactly the same as the target class.

    """

    # This function performs the docstring append on a given definition
    def do_append(target):
        # In Python 2, classes inheriting object cannot have their __doc__
        # modified after definition
        if PY2 and isclass(target) and issubclass(target, object):
            # Make a dummy class inheriting the target class
            class Target(target):
                # Perform modified append
                if target.__doc__:
                    __doc__ = join.join([target.__doc__, addendum])
                else:
                    __doc__ = addendum

            # Copy over the name and module of the target class
            Target.__name__ = target.__name__
            Target.__module__ = target.__module__
            target = Target

        # Perform normal append in all other cases
        else:
            if target.__doc__:
                target.__doc__ = join.join([target.__doc__, addendum])
            else:
                target.__doc__ = addendum

        # Return the target definition
        return(target)

    # Return decorator function
    return(do_append)


# Define custom decorator for copying docstrings from one function to another
def docstring_copy(source):
    """
    Custom decorator that allows the docstring of a function/class `source` to
    be copied to the target function/class.

    Note
    ----
    In Python 2, classes that inherit the :class:`~object` class cannot have
    their docstrings modified after it has been defined. In this case, this
    decorator makes a subclass of the target class, sets the docstring during
    class definition and returns that instead. The returned subclass is
    functionally exactly the same as the target class.

    """

    # This function performs the docstring copy on a given definition
    def do_copy(target):
        # Check if source has a docstring
        if source.__doc__:
            # In Python 2, classes inheriting object cannot have their __doc__
            # modified after definition
            if PY2 and isclass(target) and issubclass(target, object):
                # Make a dummy class inheriting the target class
                class Target(target):
                    # Perform modified copy
                    __doc__ = source.__doc__

                # Copy over the name and module of the target class
                Target.__name__ = target.__name__
                Target.__module__ = target.__module__
                target = Target

            # Perform normal copy in all other cases
            else:
                target.__doc__ = source.__doc__

        # Return the target definition
        return(target)

    # Return decorator function
    return(do_copy)


# Define custom decorator for substituting strings into a function's docstring
def docstring_substitute(*args, **kwargs):
    """
    Custom decorator that allows either given positional arguments `args` or
    keyword arguments `kwargs` to be substituted into the docstring of the
    target function/class.

    Note
    ----
    In Python 2, classes that inherit the :class:`~object` class cannot have
    their docstrings modified after it has been defined. In this case, this
    decorator makes a subclass of the target class, sets the docstring during
    class definition and returns that instead. The returned subclass is
    functionally exactly the same as the target class.

    """

    # Check if solely args or kwargs were provided
    if len(args) and len(kwargs):
        raise InputError("Either only positional or keyword arguments are "
                         "allowed!")
    else:
        params = args or kwargs

    # This function performs the docstring substitution on a given definition
    def do_substitution(target):
        # Check if target has a docstring that can be substituted to
        if target.__doc__:
            # In Python 2, classes inheriting object cannot have their __doc__
            # modified after definition
            if PY2 and isclass(target) and issubclass(target, object):
                # Make a dummy class inheriting the target class
                class Target(target):
                    # Perform modified substitution
                    __doc__ = target.__doc__ % (params)

                # Copy over the name and module of the target class
                Target.__name__ = target.__name__
                Target.__module__ = target.__module__
                target = Target

            # Perform normal substitution in all other cases
            else:
                target.__doc__ = target.__doc__ % (params)

        # Raise error if target has no docstring
        else:
            raise InputError("Target has no docstring available for "
                             "substitutions!")

        # Return the target definition
        return(target)

    # Return decorator function
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
            return(False)
    else:
        return(True)


# Function for converting a string sequence to a sequence of elements
def convert_str_seq(seq):
    """
    Converts a provided sequence to a string, removes all auxiliary characters
    from it, splits it up into individual elements and converts all elements
    back to integers, floats and/or strings.

    The auxiliary characters are given by :obj:`~aux_char_set`. One can add,
    change and remove characters from the set if required. If one wishes to
    keep an auxiliary character that is in `seq`, it must be escaped by a
    backslash (note that backslashes themselves also need to be escaped).

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

    # Convert sequence to a list of individual characters
    seq = list(str(seq))

    # Process all backslashes
    for index, char in enumerate(seq):
        # If char is a backslash
        if(char == '\\'):
            # If this backslash is escaped, skip
            if(index != 0 and seq[index-1] is None):
                pass
            # Else, if this backslash escapes a character, replace by None
            elif(index != len(seq)-1 and seq[index+1] in aux_char_set):
                seq[index] = None

    # Remove all unwanted characters from the string, except those escaped
    for char in aux_char_set:
        # Set the search index
        index = 0

        # Keep looking for the specified character
        while True:
            # Check if the character can be found in seq or break if not
            try:
                index = seq.index(char, index)
            except ValueError:
                break

            # If so, remove it if it was not escaped
            if(index == 0 or seq[index-1] is not None):
                seq[index] = '\n'
            # If it was escaped, remove None instead
            else:
                seq[index-1] = ''

            # Increment search index by 1
            index += 1

    # Convert seq back to a single string
    seq = ''.join(seq)

    # Split sequence up into elements
    seq = seq.split('\n')

    # Remove all empty strings
    while True:
        try:
            seq.remove('')
        except ValueError:
            break

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


# List/set of auxiliary characters to be used in convert_str_seq()
aux_char_set = set(['(', ')', '[', ']', ',', "'", '"', '|', '/', '\\', '{',
                    '}', '<', '>', '´', '¨', '`', '?', '!', '%', ':', ';', '=',
                    '$', '~', '#', '@', '^', '&', '*', '“', '’', '”', '‘',
                    ' '])


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
def get_outer_frame(func):
    """
    Checks whether or not the calling function contains an outer frame
    corresponding to `func` and returns it if so. If this frame cannot be
    found, returns *None* instead.

    Parameters
    ----------
    func : function
        The function or method whose frame must be located in the outer frames.

    Returns
    -------
    outer_frame : frame or None
        The requested outer frame if it was found, or *None* if it was not.

    """

    # If name is a function, obtain its name and module name
    if isfunction(func):
        name = func.__name__
        module_name = func.__module__
    # Else, if name is a method, obtain its name and class name
    elif ismethod(func):
        name = func.__name__
        class_name = func.__self__.__class__.__name__
    # Else, raise error
    else:
        raise InputError("Input argument 'func' must be a callable function or"
                         " method!")

    # Obtain the caller's frame
    caller_frame = currentframe().f_back

    # Loop over all outer frames
    for frame_info in getouterframes(caller_frame):
        # Check if frame has the correct name
        if(frame_info[3] == name):
            # If func is a function, return if module name is also correct
            if(isfunction(func) and
               frame_info[0].f_globals['__name__'] == module_name):
                return(frame_info[0])

            # Else, return frame if class name is also correct
            elif(frame_info[0].f_locals['self'].__class__.__name__ ==
                 class_name):
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
            cm.register_cmap(cmap=cmap)
            setattr(cm, cm_name, cmap)
            cm.register_cmap(cmap=cmap_r)
            setattr(cm, cm_name+'_r', cmap_r)
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
