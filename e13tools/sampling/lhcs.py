# -*- coding: utf-8 -*-

"""
LHCS
====
Provides a Latin Hypercube Sampling method.

This code is an adaptation of the original code published by Abraham Lee in the
pyDOE-package (version: 0.3.8). URL: <https://github.com/tisimst/pyDOE>

"""
# %% IMPORTS
import numpy as np


# %% FUNCTIONS
def lhs(n_val, n_sam, val_rng=None, criterion=None, iterations=1000):
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
    criterion : string or None. Default: None
        Allowed are 'center'/'c', 'maximin'/'m', 'centermaximin'/'cm' and
        'correlate'/'corr' for specific methods or *None* for randomized.
        If `n_sam` == 1, `criterion` is set to the closest corresponding
        method.
    iterations : int. Default: 1000
        Number of iterations for the maximin and correlations algorithms.

    Returns
    ------
    sam_set : 2D array_like
        Sample set array of shape [`n_sam`, `n_val`].

    """

    # Check if valid 'criterion' is given
    if criterion is not None:
        if not criterion.lower() in ('center', 'c', 'maximin', 'm',
                                     'centermaximin', 'cm', 'correlate',
                                     'corr'):
            raise ValueError("Invalid value for 'criterion': %s" % (criterion))

    # Check if n_sam > 1. If not, criterion will be changed to something useful
    if(n_sam == 1 and criterion.lower() in ('centermaximin', 'cm')):
        criterion = 'center'
    elif(n_sam == 1 and criterion.lower() in ('maximin', 'm', 'correlate',
                                              'corr')):
        criterion = None

    # Pick correct lhs-method according to criterion
    if criterion is None:
        sam_set = _lhs_classic(n_val, n_sam)
    elif criterion.lower() in ('center', 'c'):
        sam_set = _lhs_center(n_val, n_sam)
    elif criterion.lower() in ('maximin', 'm'):
        sam_set = _lhs_maximin(n_val, n_sam, 'maximin', iterations)
    elif criterion.lower() in ('centermaximin', 'cm'):
        sam_set = _lhs_maximin(n_val, n_sam, 'centermaximin', iterations)
    elif criterion.lower() in ('correlate', 'corr'):
        sam_set = _lhs_correlate(n_val, n_sam, iterations)

    # If a val_rng was given, scale sam_set to this range
    if val_rng is not None:
        # If val_rng is 1D, convert it to 2D (expected for 'n_val' = 1)
        val_rng = np.atleast_2d(val_rng)

        # Check if the given val_rng is in the correct shape
        if not(np.shape(val_rng) == (n_val, 2)):
            raise ValueError("'val_rng' has incompatible shape: (%s, %s) != "
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


def _lhs_maximin(n_val, n_sam, maximin_type, iterations):
    # Initialize maximum distance variable
    d_max = 0

    # Maximize the minimum distance between points
    for i in range(iterations):
        if(maximin_type == 'maximin'):
            sam_set_try = _lhs_classic(n_val, n_sam)
        else:
            sam_set_try = _lhs_center(n_val, n_sam)

        # Calculate the distances between all points in 'sam_set_try'
        p_dist = _get_p_dist(sam_set_try)

        # If the smallest distance in this list is bigger than 'd_max', save it
        if(d_max < np.min(p_dist)):
            d_max = np.min(p_dist)
            sam_set = sam_set_try

    # Return sam_set
    return(sam_set)


def _lhs_correlate(n_val, n_sam, iterations):
    raise NotImplementedError


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

    # If sam_set is only 1D, convert it to 2D
    sam_set = np.atleast_2d(sam_set)

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