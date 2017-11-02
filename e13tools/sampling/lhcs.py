# -*- coding: utf-8 -*-

"""
LHCS
====
Provides a Latin Hypercube Sampling method.

This code is an adaptation of the original code published by Abraham Lee in the
pyDOE-package (version: 0.3.8). URL: <https://github.com/tisimst/pyDOE>

"""


# %% IMPORTS
from __future__ import division, absolute_import, print_function

from e13tools import ShapeError
import numpy as np

__all__ = ['lhs']


# %% FUNCTIONS
def lhs(n_val, n_sam, val_rng=None, criterion='random', iterations=1000,
        constraints=[[]]):
    """
    Generate a Latin Hypercube of `n_sam` samples, each with `n_val` values.

    Parameters
    ----------
    n_val : int
        The number of values in a single sample.
    n_sam : int
        The number of samples to generate.

    Optional
    --------
    val_rng : 2D array_like or None. Default: None
        Array defining the lower and upper limits of every value in a sample.
        Requires: numpy.shape(val_rng) = (`n_val`, 2).
        If *None*, output is normalized.
    criterion : string. Default: 'random'
        Allowed are 'center'/'c', 'maximin'/'m', 'centermaximin'/'cm' for
        specific methods or 'random'/'r' for randomized.
        If `n_sam` == 1, `criterion` is set to the closest corresponding
        method.
    iterations : int. Default: 1000
        Number of iterations for the maximin and correlations algorithms.
    constraints : 2D array_like. Default: [[]]
        If `constraints` is not empty and `criterion` is set to 'maximin'/'m'
        or 'centermaximin'/'cm', both `sam_set` and `sam_set` + `constraints`
        will satisfy the given criterion.

    Returns
    ------
    sam_set : 2D array_like
        Sample set array of shape [`n_sam`, `n_val`].

    Examples
    --------
    Latin Hypercube with 2 values and 5 samples:

    >>> lhs(2, 5)
    array([[ 0.33049758,  0.46107203],
           [ 0.0489103 ,  0.24679923],
           [ 0.59167567,  0.74001012],
           [ 0.78958876,  0.10264081],
           [ 0.98245615,  0.99973969]])


    Latin Hypercube with 3 values, 4 samples in a specified value range:

    >>> val_rng = [[0, 2], [1, 4], [0.3, 0.5]]
    >>> lhs(3, 4, val_rng=val_rng)
    array([[ 0.94523229,  3.66450868,  0.49843528],
           [ 1.13087439,  2.80949861,  0.31239253],
           [ 0.17403476,  2.11747903,  0.4113592 ],
           [ 1.59869101,  1.16117621,  0.38345357]])


    Latin Hypercube with 5 values and 6 centered samples:

    >>> lhs(5, 6, criterion='center')
    array([[ 0.91666667,  0.91666667,  0.75      ,  0.25      ,  0.58333333],
           [ 0.08333333,  0.58333333,  0.25      ,  0.41666667,  0.41666667],
           [ 0.58333333,  0.25      ,  0.08333333,  0.58333333,  0.25      ],
           [ 0.25      ,  0.08333333,  0.58333333,  0.91666667,  0.08333333],
           [ 0.41666667,  0.41666667,  0.41666667,  0.08333333,  0.91666667],
           [ 0.75      ,  0.75      ,  0.91666667,  0.75      ,  0.75      ]])


    Latin Hypercubes can also be created with the 'maximin' criterion:

    >>> lhs(3, 4, criterion='maximin')
    array([[ 0.13666344,  0.04666906,  0.41487346],
           [ 0.863929  ,  0.35886462,  0.99011525],
           [ 0.73400563,  0.55282619,  0.07831996],
           [ 0.26816676,  0.8538964 ,  0.71979893]])


    Finally, an existing Latin Hypercube can be provided as an additional
    constraint for calculating maximin Latin Hypercubes, if the number of
    values are equal:

    >>> cube1 = lhs(1, 2, criterion='random')
    >>> cube2 = lhs(1, 6, criterion='center')
    >>> cube = np.vstack([cube1, cube2])
    >>> cube
    array([[ 0.06495814],
           [ 0.85064614],
           [ 0.91666667],
           [ 0.41666667],
           [ 0.58333333],
           [ 0.25      ],
           [ 0.08333333],
           [ 0.75      ]])
    >>> lhs(1, 5, criterion='maximin', constraints=cube)
    array([[ 0.51435233],
           [ 0.15430434],
           [ 0.6062859 ],
           [ 0.22938232],
           [ 0.999933  ]])

    """

    # Check if valid 'criterion' is given
    if not criterion.lower() in ('center', 'c', 'maximin', 'm',
                                 'centermaximin', 'cm', 'random', 'r'):
        raise ValueError("Invalid value for 'criterion': %s" % (criterion))

    # Check the shape of 'constraints' and act accordingly
    if(np.shape(constraints)[-1] == 0):
        # If constraints is empty, there are no constraints
        constraints = None
    elif not criterion.lower() in ('maximin', 'm', 'centermaximin', 'cm'):
        # If non-compatible criterion is provided, there are no constraints
        constraints = None
    elif(np.shape(np.shape(constraints))[0] != 2):
        # If constraints is not two-dimensional, it is invalid
        raise ShapeError("Constraints must be two-dimensional!")
    elif(np.shape(constraints)[1] == n_val):
        # If constraints has the same number of values, it is valid
        constraints = _extract_sam_set(constraints, val_rng)

        # If constraints is empty after extraction, there are no constraints
        if(np.shape(constraints)[-1] == 0):
            constraints = None
    else:
        # If not empty and not right shape, it is invalid
        raise ShapeError("Constraints has incompatible number of values: "
                         "%s =! %s" % (np.shape(constraints)[1], n_val))

    # Check if n_sam > 1. If not, criterion will be changed to something useful
    if(n_sam == 1 and criterion.lower() in ('center', 'c', 'random', 'r')):
        pass
    elif(n_sam == 1 and criterion.lower() in ('centermaximin', 'cm') and
         constraints is None):
        criterion = 'center'
    elif(n_sam == 1 and criterion.lower() in ('maximin', 'm') and
         constraints is None):
        criterion = 'random'

    # Pick correct lhs-method according to criterion
    if criterion.lower() in ('random', 'r'):
        sam_set = _lhs_classic(n_val, n_sam)
    elif criterion.lower() in ('center', 'c'):
        sam_set = _lhs_center(n_val, n_sam)
    elif criterion.lower() in ('maximin', 'm'):
        sam_set = _lhs_maximin(n_val, n_sam, 'maximin', iterations,
                               constraints)
    elif criterion.lower() in ('centermaximin', 'cm'):
        sam_set = _lhs_maximin(n_val, n_sam, 'centermaximin', iterations,
                               constraints)

    # If a val_rng was given, scale sam_set to this range
    if val_rng is not None:
        # If val_rng is 1D, convert it to 2D (expected for 'n_val' = 1)
        val_rng = np.atleast_2d(val_rng)

        # Check if the given val_rng is in the correct shape
        if not(np.shape(val_rng) == (n_val, 2)):
            raise ShapeError("'val_rng' has incompatible shape: (%s, %s) != "
                             "(%s, %s)" % (np.shape(val_rng)[0],
                                           np.shape(val_rng)[1], n_val, 2))

        # Scale sam_set according to val_rng
        sam_set = val_rng[:, 0]+sam_set*(val_rng[:, 1]-val_rng[:, 0])

    return(sam_set)


def _lhs_classic(n_val, n_sam):
    # Generate the equally spaced intervals/bins
    bins = np.linspace(0, 1, n_sam+1)

    # Obtain lower and upper bounds of bins
    bins_low = bins[0:n_sam]
    bins_high = bins[1:n_sam+1]

    # Generate random points
    rgn = np.random.rand(n_sam, n_val)

    # Distribute randomly generated numbers over bins
    for i in range(n_val):
        rgn[:, i] = bins_low+rgn[:, i]*(bins_high-bins_low)

    # Pair values randomly together to obtain random samples
    sam_set = np.zeros_like(rgn)
    for i in range(n_val):
        order = np.random.permutation(range(n_sam))
        sam_set[:, i] = rgn[order, i]

    # Return sam_set
    return(sam_set)


def _lhs_center(n_val, n_sam):
    # Generate the equally spaced intervals/bins
    bins = np.linspace(0, 1, n_sam+1)

    # Obtain lower and upper bounds of bins
    bins_low = bins[0:n_sam]
    bins_high = bins[1:n_sam+1]

    # Capture centers of every bin
    center_num = (bins_low+bins_high)/2

    # Pair values randomly together to obtain random samples
    sam_set = np.zeros([n_sam, n_val])
    for i in range(n_val):
        sam_set[:, i] = np.random.permutation(center_num)

    # Return sam_set
    return(sam_set)


def _lhs_maximin(n_val, n_sam, maximin_type, iterations, constraints):
    # Initialize maximum distance variable
    d_max = 0

    # Maximize the minimum distance between points
    for i in range(iterations):
        if(maximin_type == 'maximin'):
            sam_set_try = _lhs_classic(n_val, n_sam)
        else:
            sam_set_try = _lhs_center(n_val, n_sam)

        # If constraints is not None, then it needs to be added to sam_set_try
        if constraints is not None:
            sam_set_try_full = np.vstack([sam_set_try, constraints])

            # Calculate the distances between all points in 'sam_set_try'
            p_dist = _get_p_dist(sam_set_try_full)
        else:
            p_dist = _get_p_dist(sam_set_try)

        # If the smallest distance in this list is bigger than 'd_max', save it
        if(d_max < np.min(p_dist)):
            d_max = np.min(p_dist)
            sam_set = sam_set_try

    # Return sam_set
    return(sam_set)


def _get_p_dist(sam_set):
    """
    Calculate the pair-wise point distances of a given sample set `sam_set`.

    Parameters
    ----------
    sam_set : 2D array_like
        Sample set array of shape [`n_sam`, `n_val`].

    Returns
    -------
    p_dist_vec : 1D array_like
        Vector containing all pair-wise point distances.

    """

    # Obtain number of values in number of samples
    n_sam, n_val = np.shape(sam_set)

    # Initialize point distance vector
    p_dist_vec = np.zeros(int(0.5*n_sam*(n_sam-1)))

    # Calculate pair-wise point distances
    k = 0
    for i in range(n_sam-1):
        for j in range(i+1, n_sam):
            p_dist_vec[k] = np.sqrt(sum(pow(sam_set[i, :]-sam_set[j, :], 2)))
            k += 1

    # Return point distance vector
    return(p_dist_vec)


def _extract_sam_set(sam_set, val_rng):
    """
    Extracts the samples from `sam_set` that are within the given value
    ranges `val_rng`. Also extracts the two samples that are the closest to the
    given value ranges, but are outside of it.

    Parameters
    ----------
    sam_set : 2D array_like
        Sample set containing the samples that require extraction.
    val_rng : 2D array_like or None
        Array defining the lower and upper limits of every value in a sample.
        If *None*, output is normalized.

    Returns
    -------
    ext_sam_set : 2D array_like
        Sample set containing the extracted samples.

    """

    # Obtain number of values in number of samples
    n_sam, n_val = np.shape(sam_set)

    # Check if val_rng is given. If not, set it to default range
    if val_rng is None:
        val_rng = np.zeros([n_val, 2])
        val_rng[:, 1] = 1
    else:
        # If val_rng is 1D, convert it to 2D (expected for 'n_val' = 1)
        val_rng = np.atleast_2d(val_rng)

    # Scale all samples to the value range [0, 1]
    sam_set = ((sam_set-val_rng[:, 0])/(val_rng[:, 1]-val_rng[:, 0]))

    # Create empty array of valid samples
    ext_sam_set = np.array([[]])

    # Create empty arrays of samples that are just outside the val_rng
    upper_sam = np.array([[]])
    upper_dist_min = np.infty
    lower_sam = np.array([[]])
    lower_dist_min = np.infty

    # Check which samples are within val_rng or just outside of it
    for i in range(n_sam):
        # If a sample is within the value range, save it
        if((0 <= sam_set[i, :]).any() and (sam_set[i, :] <= 1).any()):
            if(np.shape(ext_sam_set)[1] == 0):
                ext_sam_set = np.atleast_2d(sam_set[i])
            else:
                ext_sam_set = np.vstack([ext_sam_set, sam_set[i, :]])
        # If a sample is just outside the value range, save it if it is closer
        # than the one that is currently saved
        elif((sam_set[i, :] < 0).all()):
            lower_dist = np.sqrt(sum(pow(sam_set[i, :] -
                                         np.zeros_like(sam_set[i, :]), 2)))
            if(lower_dist_min > lower_dist):
                lower_dist_min = lower_dist
                lower_sam = sam_set[i, :]
        # Do the same thing for the upper limit of the value range as well
        elif((1 < sam_set[i, :]).all()):
            upper_dist = np.sqrt(sum(pow(sam_set[i, :] -
                                         np.ones_like(sam_set[i, :]), 2)))
            if(upper_dist_min > upper_dist):
                upper_dist_min = upper_dist
                upper_sam = sam_set[i, :]

    # If lower_sam and/or upper_sam are not empty, add them to sam_set
    # Take into account that sam_set might still be empty
    if(np.shape(lower_sam) != np.shape([[]]) and
       np.shape(ext_sam_set) == np.shape([[]])):
        ext_sam_set = np.atleast_2d(lower_sam)
    elif(np.shape(lower_sam) != np.shape([[]]) and
         np.shape(ext_sam_set) != np.shape([[]])):
        ext_sam_set = np.vstack([ext_sam_set, np.atleast_2d(lower_sam)])
    if(np.shape(upper_sam) != np.shape([[]]) and
       np.shape(ext_sam_set) == np.shape([[]])):
        ext_sam_set = np.atleast_2d(upper_sam)
    elif(np.shape(upper_sam) != np.shape([[]]) and
         np.shape(ext_sam_set) != np.shape([[]])):
        ext_sam_set = np.vstack([ext_sam_set, np.atleast_2d(upper_sam)])

    # Return sam_set
    return(ext_sam_set)
