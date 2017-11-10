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
    Method for choosing the 'best' Latin Hypercube Design depends on the
    `criterion` that is used.

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
        Allowed are 'center'/'c', 'maximin'/'m', 'centermaximin'/'cm',
        'correlation'/'corr', 'maximincorr'/'multi' for specific methods or
        'random'/'r' for randomized.
        If `n_sam` == 1 or `n_val == 1`, `criterion` is set to the closest
        corresponding method if necessary.
    iterations : int. Default: 1000
        Number of iterations for the maximin and correlation algorithms.
    constraints : 2D array_like. Default: [[]]
        If `constraints` is not empty and `criterion` is set to any maximin or
        correlation method, both `sam_set` and `sam_set` + `constraints` will
        satisfy the given criterion.

    Returns
    -------
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


    Latin Hypercubes can also be created with the 'maximin' criterion, which
    tries to maximize the minimum distance between any pair of samples:

    >>> lhs(3, 4, criterion='maximin')
    array([[ 0.13666344,  0.04666906,  0.41487346],
           [ 0.863929  ,  0.35886462,  0.99011525],
           [ 0.73400563,  0.55282619,  0.07831996],
           [ 0.26816676,  0.8538964 ,  0.71979893]])


    Additionally, Latin Hypercubes can be made with the 'correlation'
    criterion, which instead tries to minimize the cross-correlation between
    any pair of samples:

    >>> lhs(3, 4, criterion='correlation')
    array([[ 0.59248275,  0.500076  ,  0.33836536],
           [ 0.89500233,  0.04831665,  0.6891592 ],
           [ 0.08223136,  0.87907617,  0.8142178 ],
           [ 0.30950514,  0.45548182,  0.09802957]])


    If one wants to combine both the 'maximin' and the 'correlation'
    criterions, the 'multi' criterion can be used to generate a Latin Hypercube
    that tries to maximize the minimum distance and minimize the
    cross-correlation between any pair of samples simultaneously:

    >>> lhs(3, 4, criterion='multi')
    array([[ 0.03612175,  0.46903638,  0.02233707],
           [ 0.88992134,  0.57888616,  0.2588855 ],
           [ 0.55147305,  0.04057838,  0.95252719],
           [ 0.31951143,  0.99777721,  0.67487922]])


    Finally, an existing Latin Hypercube can be provided as an additional
    constraint for calculating maximin or correlation Latin Hypercubes, if the
    number of values are equal (using `n_val` = 1 shows its impact clearly):

    >>> import numpy as np
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
                                 'centermaximin', 'cm', 'correlation', 'corr',
                                 'maximincorr', 'multi', 'random', 'r'):
        raise ValueError("Invalid value for 'criterion': %s" % (criterion))

    # Check the shape of 'constraints' and act accordingly
    if(np.shape(constraints)[-1] == 0):
        # If constraints is empty, there are no constraints
        constraints = None
    elif not criterion.lower() in ('maximin', 'm', 'centermaximin', 'cm',
                                   'correlation', 'corr', 'maximincorr',
                                   'multi'):
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

    # Check for cases in which some criterions make no sense
    # If so, criterion will be changed to something useful
    if(n_sam == 1 and criterion.lower() in ('center', 'c', 'random', 'r')):
        pass
    elif((n_val == 1 or n_sam == 1) and
         criterion.lower() in ('centermaximin', 'cm')):
        criterion = 'center'
    elif(n_sam == 1 and
         criterion.lower() in ('maximin', 'm', 'correlation', 'corr',
                               'maximincorr', 'multi') and
         constraints is None):
        criterion = 'random'
    elif(n_val <= 2 and criterion.lower() in ('correlation', 'corr')):
        criterion = 'random'
    elif(n_val == 1 and criterion.lower() in ('maximincorr', 'multi')):
        criterion = 'maximin'

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
    elif criterion.lower() in ('correlation', 'corr'):
        sam_set = _lhs_correlation(n_val, n_sam, iterations, constraints)
    elif criterion.lower() in ('maximincorr', 'multi'):
        sam_set = _lhs_maximincorr(n_val, n_sam, iterations, constraints)

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
        else:
            sam_set_try_full = sam_set_try

        # Calculate the distances between all points
        p_dist = _get_p_dist(sam_set_try_full)

        # If the smallest distance in this list is bigger than 'd_max', save it
        d_min = np.min(p_dist)
        if(d_max < d_min):
            d_max = d_min
            sam_set = sam_set_try

    # Return sam_set
    return(sam_set)


def _lhs_correlation(n_val, n_sam, iterations, constraints):
    # Initialize minimum correlation variable
    c_min = np.infty

    # Minimize cross-correlation between samples
    for i in range(iterations):
        sam_set_try = _lhs_classic(n_val, n_sam)

        # Calculate the correlation between all points
        R_corr = np.corrcoef(sam_set_try, constraints)

        # If the highest cross-correlation is lower than 'c_min', save it
        c_max = np.max(np.abs(R_corr-np.eye(len(R_corr))))
        if(c_max < c_min):
            c_min = c_max
            sam_set = sam_set_try

    # Return sam_set
    return(sam_set)


def _lhs_maximincorr(n_val, n_sam, iterations, constraints):
    # Initialize maximum distance and minimum correlation variables
    d_max = 0
    c_min = np.infty

    # Maximize the minimum distance and minimize the cross-correlation between
    # samples
    for i in range(iterations):
        sam_set_try = _lhs_classic(n_val, n_sam)

        # If constraints is not None, then it needs to be added to sam_set_try
        if constraints is not None:
            sam_set_try_full = np.vstack([sam_set_try, constraints])
        else:
            sam_set_try_full = sam_set_try

        # Calculate the distances and correlation between all points
        p_dist = _get_p_dist(sam_set_try_full)
        R_corr = np.corrcoef(sam_set_try, constraints)

        # If the smallest distance is bigger than 'd_max' and the highest
        # cross-correlation is lower than 'corr_min', save it
        d_min = np.min(p_dist)
        c_max = np.max(np.abs(R_corr-np.eye(len(R_corr))))
        if(d_max < d_min and c_max < c_min):
            d_max = d_min
            c_max = c_min
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

    # Create lower and upper limits of the hypercube containing samples that
    # can influence the created hypercube
    lower_lim = 0-np.sqrt(n_val)
    upper_lim = 1+np.sqrt(n_val)

    # Check which samples are within val_rng or just outside of it
    for i in range(n_sam):
        # If a sample is within the outer hypercube, save it
        if(((lower_lim <= sam_set[i, :])*(sam_set[i, :] <= upper_lim)).all()):
            if(np.shape(ext_sam_set)[1] == 0):
                ext_sam_set = np.atleast_2d(sam_set[i])
            else:
                ext_sam_set = np.vstack([ext_sam_set, sam_set[i, :]])

    # Return sam_set
    return(ext_sam_set)


# %% DOCTEST
if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.SKIP)
