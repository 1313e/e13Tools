# -*- coding: utf-8 -*-

"""
Math Core
=========
Provides a collection of functions that are core to **Math** and are imported
automatically.


Available functions
-------------------
:func:`~diff`
    Calculates the pair-wise differences between inputs `array1` and `array2`
    along the given axis.

:func:`~gcd`
    Returns the greatest common divisor of the provided sequence of integers.

:func:`~is_PD`
    Checks if `matrix` is positive-definite or not, by using the
    :func:`~np.linalg.cholesky` function. It is required for `matrix` to be
    Hermitian.

:func:`~lcm`
    Returns the least common multiple of the provided sequence of integers.
    If at least one integer is zero, the output will also be zero.

:func:`~nCr`
    For a given set S of `n` elements, returns the number of unordered
    arrangements ("combinations") of length `r` one can make with S.
    Returns zero if `r` > `n` and `repeat` is *False*.

:func:`~nearest_PD`
    Find the nearest positive-definite matrix to the input `matrix`.

:func:`~nPr`
    For a given set S of `n` elements, returns the number of ordered
    arrangements ("permutations") of length `r` one can make with S.
    Returns zero if `r` > `n` and `repeat` is *False*.

:func:`~rot90`
    Rotates the given `array` by 90 degrees around the point `rot_axis` in the
    given `axes`. This function is different from NumPy's :func:`~numpy.rot90`
    function in that every column (2nd axis) defines a different dimension
    instead of every individual axis.

:func:`~sort_2D`
    Sorts a 2D `array` in a given `axis` in the specified `order`. This
    function is different from NumPy's :func:`~sort` function in that it sorts
    in a given axis rather than along it, and the order can be given as
    integers rather than field strings.

:func:`~transposeC`
    Returns the (conjugate) transpose of the input `array`.

"""


# %% IMPORTS
from __future__ import absolute_import, division, print_function

from e13tools import InputError, ShapeError
from math import factorial
from functools import reduce
import numpy as np
from numpy.linalg import cholesky, eigvals, LinAlgError, norm, svd

__all__ = ['diff', 'gcd', 'is_PD', 'lcm', 'nCr', 'nearest_PD', 'nPr', 'rot90',
           'sort_2D', 'transposeC']


# %% FUNCTIONS
def diff(array1, array2=None, axis=0, flatten=True):
    """
    Calculates the pair-wise differences between inputs `array1` and `array2`
    along the given `axis`.

    Parameters
    ----------
    array1 : array_like
        One of the inputs used to calculate the pair-wise differences.

    Optional
    --------
    array2 : array_like or None. Default: None
        The other input used to calculate the pair-wise differences.
        If *None*, `array2` is equal to `array1`.
        If not *None*, the length of all axes except `axis` must be equal for
        both arrays.
    axis : int. Default: 0
        Along which axis to calculate the pair-wise differences. Default is
        along the first axis. A negative value counts from the last to the
        first axis.
    flatten : bool. Default: True
        If `array2` is *None*, whether or not to calculate all pair-wise
        differences.
        If *True*, a flattened array containing all above-diagonal pair-wise
        differences is returned. This is useful if only off-diagonal terms are
        required and the sign is not important.
        If *False*, an array with all pair-wise differences is returned.

    Returns
    -------
    diff_array : :obj:`~numpy.ndarray` object
        Depending on the input parameters, an array with n dimensions
        containing the pair-wise differences between `array1` and `array2`
        along the given `axis`.

    Examples
    --------
    Using two matrices returns the pair-wise differences in row-vectors:

        >>> mat1 = np.array([[1, 2, 3], [4, 5, 6]])
        >>> mat2 = np.array([[4, 5, 6], [7, 8, 9]])
        >>> diff(mat1, mat2)
        array([[[-3., -3., -3.],
                [-6., -6., -6.]],
        <BLANKLINE>
               [[ 0.,  0.,  0.],
                [-3., -3., -3.]]])


    Setting `axis` to 1 returns the pair-wise differences in column-vectors:

        >>> mat1 = np.array([[1, 2, 3], [4, 5, 6]])
        >>> mat2 = np.array([[4, 5, 6], [7, 8, 9]])
        >>> diff(mat1, mat2, axis=1)
        array([[[-3., -3.],
                [-4., -4.],
                [-5., -5.]],
        <BLANKLINE>
               [[-2., -2.],
                [-3., -3.],
                [-4., -4.]],
        <BLANKLINE>
               [[-1., -1.],
                [-2., -2.],
                [-3., -3.]]])


    Only using a single matrix returns the pair-wise differences in row-vectors
    in that matrix (either flattened or not):

        >>> mat = np.array([[1, 2, 3], [4, 5, 6]])
        >>> diff(mat, flatten=True)
        array([[-3., -3., -3.]])
        >>> diff(mat, flatten=False)
        array([[[ 0.,  0.,  0.],
                [-3., -3., -3.]],
        <BLANKLINE>
               [[ 3.,  3.,  3.],
                [ 0.,  0.,  0.]]])


    Using a matrix and a vector returns the pair-wise differences in
    row-vectors:

        >>> mat = np.array([[1, 2, 3], [4, 5, 6]])
        >>> vec = np.array([7, 8, 9])
        >>> diff(mat, vec)
        array([[-6, -6, -6],
               [-3, -3, -3]])


    Using two vectors returns the pair-wise differences in scalars:

        >>> vec1 = np.array([1, 2, 3])
        >>> vec2 = np.array([4, 5, 6])
        >>> diff(vec1, vec2)
        array([[-3., -4., -5.],
               [-2., -3., -4.],
               [-1., -2., -3.]])

    """

    # If array2 is not provided, both arrays are the same
    if array2 is None:
        # Make sure that input is a numpy array
        array1 = np.array(array1)

        # Check if a scalar has been provided and act accordingly
        if(array1.ndim == 0):
            return(0)

        # Swap axes in array to put the given axis as the first axis
        try:
            array1 = np.moveaxis(array1, axis, 0).copy()
        except Exception as error:
            raise InputError("Input argument 'axis' is invalid (%s)!"
                             % (error))
        else:
            # Obtain the dimensionality and axis-length
            n_dim = array1.ndim
            len_axis = array1.shape[0]

        # If only unique pair-wise differences are requested
        if flatten:
            # Obtain the shape of the resulting array and initialize it
            n_diff = len_axis*(len_axis-1)//2
            if(n_dim == 1):
                diff_shape = [n_diff]
            else:
                diff_shape = np.concatenate([[n_diff], array1.shape[1:n_dim]])
            diff_array = np.zeros(diff_shape)

            # Initialize empty variable holding the distance in index of last i
            dist = 0

            # Fill array
            for i in range(len_axis):
                diff_array[dist:dist+len_axis-i-1] = array1[i]-array1[i+1:]
                dist += len_axis-i-1

            # Return it
            return(diff_array)

        # If all difference are requested
        else:
            # Obtain the shape of the resulting array and initialize it
            diff_shape = np.concatenate([[len_axis], array1.shape])
            diff_array = np.zeros(diff_shape)

            # Fill array
            for i in range(len_axis):
                diff_array[i] = array1[i]-array1

            # Return it
            return(diff_array)

    # If array2 is provided, both arrays are different
    else:
        # Make sure that inputs are numpy arrays
        array1 = np.array(array1)
        array2 = np.array(array2)

        # Get number of dimensions
        n_dim1 = array1.ndim
        n_dim2 = array2.ndim

        # Check if both arrays are scalars and act accordingly
        if(n_dim1 == n_dim2 == 0):
            return(array1-array2)

        # If both arrays have the same number of dimensions
        if(n_dim1 == n_dim2):
            # Swap axes in arrays to put the given axis as the first axis
            try:
                array1 = np.moveaxis(array1, axis, 0).copy()
                array2 = np.moveaxis(array2, axis, 0).copy()
            except Exception as error:
                raise InputError("Input argument 'axis' is invalid (%s)!"
                                 % (error))
            else:
                # Obtain axis-length
                len_axis1 = array1.shape[0]

            # Check if the length of all other axes are the same
            if(array1.shape[1:n_dim1] != array2.shape[1:n_dim2]):
                raise ShapeError("Input arguments 'array1' and 'array2' do not"
                                 " have the same axes lengths: %s != %s"
                                 % (array1.shape[1:n_dim1],
                                    array2.shape[1:n_dim2]))

            # Obtain the shape of the resulting array and initialize it
            diff_shape = np.concatenate([[len_axis1], array2.shape])
            diff_array = np.zeros(diff_shape)

            # Fill array
            for i in range(len_axis1):
                diff_array[i] = array1[i]-array2

            # Return it
            return(diff_array)

        # If the arrays have different number of dimensions
        else:
            # Swap axes in the bigger array to put the given axis as first axis
            if(n_dim1 > n_dim2):
                try:
                    array1 = np.moveaxis(array1, axis, 0).copy()
                except Exception as error:
                    raise InputError("Input argument 'axis' is invalid (%s)!"
                                     % (error))

                # Check if the length of all other axes are the same
                if(array1.shape[1:n_dim1] != array2.shape):
                    raise ShapeError("Input arguments 'array1' and 'array2' do"
                                     " not have the same axes lengths: %s != "
                                     "%s"
                                     % (array1.shape[1:n_dim1], array2.shape))
                else:
                    # Return difference array
                    return(array1-array2)
            else:
                try:
                    array2 = np.moveaxis(array2, axis, 0).copy()
                except Exception as error:
                    raise InputError("Input argument 'axis' is invalid (%s)!"
                                     % (error))

                # Check if the length of all other axes are the same
                if(array1.shape != array2.shape[1:n_dim2]):
                    raise ShapeError("Input arguments 'array1' and 'array2' do"
                                     " not have the same axes lengths: %s != "
                                     "%s"
                                     % (array1.shape, array2.shape[1:n_dim2]))
                else:
                    # Return difference array
                    return(array1-array2)


def gcd(seq):
    """
    Returns the greatest common divisor of the provided sequence of integers.

    Parameters
    ----------
    seq : 1D array_like of int
        Integers to calculate the greatest common divisor for.

    Returns
    -------
    gcd : int
        Greatest common divisor of input integers.

    Example
    -------
    >>> gcd([18, 60, 72, 138])
    6

    See also
    --------
    - :func:`~e13tools.math.core.gcd_single`: Greatest common divisor for two \
        integers.
    - :func:`~e13tools.math.lcm`: Least common multiple for sequence of \
        integers.
    - :func:`~e13tools.math.core.lcm_single`: Least common multiple for two \
        integers.

    """

    return(reduce(lambda a, b: gcd_single(a, b), seq))


def gcd_single(a, b):
    """
    Returns the greatest common divisor of the integers `a` and `b` using
    Euclid's Algorithm [1]_.

    Parameters
    ----------
    a, b : int
        The two integers to calculate the greatest common divisor for.

    Returns
    -------
    gcd : int
        Greatest common divisor of `a` and `b`.

    Notes
    -----
    The calculation of the greatest common divisor uses Euclid's Algorithm [1]_
    with Lamé's improvements.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Euclidean_algorithm

    Example
    -------
    >>> gcd_single(42, 56)
    14

    See also
    --------
    - :func:`~e13tools.math.gcd`: Greatest common divisor for sequence of \
        integers.
    - :func:`~e13tools.math.lcm`: Least common multiple for sequence of \
        integers.
    - :func:`~e13tools.math.core.lcm_single`: Least common multiple for two \
        integers.

    """

    while(b):
        a, b = b, a % b
    return(a)


def is_PD(matrix):
    """
    Checks if `matrix` is positive-definite or not, by using the
    :func:`~np.linalg.cholesky` function. It is required for `matrix` to be
    Hermitian.

    Parameters
    ----------
    matrix : 2D array_like
        Matrix that requires checking.

    Returns
    -------
    out: bool
        *True* if `matrix` is positive-definite, *False* if it is not.

    Examples
    --------
    Using a real matrix that is positive-definite (like the identity matrix):

        >>> matrix = np.eye(3)
        >>> matrix
        array([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]])
        >>> is_PD(matrix)
        True


    Using a real matrix that is not symmetric (Hermitian):

        >>> matrix = np.array([[1, 2], [3, 4]])
        >>> matrix
        array([[1, 2],
               [3, 4]])
        >>> is_PD(matrix)
        Traceback (most recent call last):
            ...
        ValueError: Input argument 'matrix' must be Hermitian!


    Using a complex matrix that is positive-definite:

        >>> matrix = np.array([[4, 1.5+1j], [1.5-1j, 3]])
        >>> matrix
        array([[ 4.0+0.j,  1.5+1.j],
               [ 1.5-1.j,  3.0+0.j]])
        >>> is_PD(matrix)
        True

    See also
    --------
    - :func:`~e13tools.math.nearest_PD`: Find the nearest positive-definite \
        matrix to the input `matrix`.

    """

    # Make sure that matrix is a numpy array
    matrix = np.array(matrix)

    # Check if input is a matrix
    if(matrix.ndim != 2):
        raise ShapeError("Input argument 'matrix' must be two-dimensional!")
    else:
        rows, columns = matrix.shape

    # Check if matrix is a square
    if(rows != columns):
        raise ShapeError("Input argument 'matrix' has shape [%s, %s]. 'matrix'"
                         " must be a square matrix!" % (rows, columns))

    # Check if matrix is Hermitian
    if not((transposeC(matrix) == matrix).all()):
        raise ValueError("Input argument 'matrix' must be Hermitian!")

    # Try to use Cholesky on matrix. If it fails,
    try:
        cholesky(matrix)
    except LinAlgError:
        return(False)
    else:
        return(True)


def lcm(seq):
    """
    Returns the least common multiple of the provided sequence of integers.
    If at least one integer is zero, the output will also be zero.

    Parameters
    ----------
    seq : 1D array_like of int
        Integers to calculate the least common multiple for.

    Returns
    -------
    lcm : int
        Least common multiple of input integers.

    Example
    -------
    >>> lcm([8, 9, 21])
    504

    See also
    --------
    - :func:`~e13tools.math.gcd`: Greatest common divisor for sequence of \
        integers.
    - :func:`~e13tools.math.core.gcd_single`: Greatest common divisor for two \
        integers.
    - :func:`~e13tools.math.core.lcm_single`: Least common multiple for two \
        integers.

    """

    return(reduce(lcm_single, seq))


def lcm_single(a, b):
    """
    Returns the least common multiple of the integers `a` and `b`.
    If at least one integer is zero, the output will also be zero.

    Parameters
    ----------
    a, b : int
        The two integers to calculate the least common multiple for.

    Returns
    -------
    lcm : int
        Least common multiple of `a` and `b`.

    Notes
    -----
    The least common multiple of two given integers :math:`a` and :math:`b` is
    given by

        .. math:: \\mathrm{lcm}(a, b)=\\frac{|a\\cdot b|}{\\mathrm{gcd}(a, b)},

    which can also be written as

        .. math:: \\mathrm{lcm}(a, b)=\\frac{|a|}{\\mathrm{gcd}(a, b)}\\cdot \
            |b|,

    with :math:`\mathrm{gcd}` being the greatest common divisor.

    Example
    -------
    >>> lcm_single(6, 21)
    42

    See also
    --------
    - :func:`~e13tools.math.gcd`: Greatest common divisor for sequence of \
        integers.
    - :func:`~e13tools.math.core.gcd_single`: Greatest common divisor for two \
        integers.
    - :func:`~e13tools.math.lcm`: Least common multiple for sequence of \
        integers.

    """

    return(0 if(a == 0 or b == 0) else (abs(a)//gcd_single(a, b))*abs(b))


def nCr(n, r, repeat=False):
    """
    For a given set S of `n` elements, returns the number of unordered
    arrangements ("combinations") of length `r` one can make with S.
    Returns zero if `r` > `n` and `repeat` is *False*.

    Parameters
    ----------
    n : int
        Number of elements in the set S.
    r : int
        Number of elements in the sub-set of set S.

    Optional
    --------
    repeat : bool. Default: False
        If *False*, each element in S can only be chosen once.
        If *True*, they can be chosen more than once.

    Returns
    -------
    n_comb : int
        Number of "combinations" that can be made with S.

    Examples
    --------
    >>> nCr(4, 2)
    6


    >>> nCr(4, 2, repeat=True)
    10


    >>> nCr(2, 4, repeat=True)
    5


    >>> nCr(2, 4)
    0

    See also
    --------
    - :func:`~e13tools.math.nPr`: Returns the number of ordered arrangements.

    """

    # Check if repeat is True or not and act accordingly
    if(r == 0):
        return(1)
    elif(r == 1):
        return(n)
    elif repeat:
        return(factorial(n+r-1)//(factorial(r)*factorial(n-1)))
    elif(r == n):
        return(1)
    elif(r > n):
        return(0)
    else:
        return(factorial(n)//(factorial(r)*factorial(n-r)))


def nearest_PD(matrix):
    """
    Find the nearest positive-definite matrix to the input `matrix`.

    Parameters
    ----------
    matrix : 2D array_like
        Input matrix that requires its nearest positive-definite variant.

    Returns
    -------
    mat_PD : 2D :obj:`~numpy.ndarray` object
        The nearest positive-definite matrix to the input `matrix`.

    Notes
    -----
    This is a Python port of John D'Errico's *nearestSPD* code [1]_, which is a
    MATLAB implementation of Higham (1988) [2]_.

    According to Higham (1988), the nearest positive semi-definite matrix in
    the Frobenius norm to an arbitrary real matrix :math:`A` is shown to be

        .. math:: \\frac{B+H}{2},

    with :math:`H` being the symmetric polar factor of

        .. math:: B=\\frac{A+A^T}{2}.

    On page 2, the author mentions that all matrices :math:`A` are assumed to
    be real, but that the method can be very easily extended to the complex
    case. This can indeed be done easily by taking the conjugate transpose
    instead of the normal transpose in the formula on the above.

    References
    ----------
    .. [1] \
        https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    .. [2] N.J. Higham, "Computing a Nearest Symmetric Positive Semidefinite
           Matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

    Examples
    --------
    Requesting the nearest PD variant of a matrix that is already PD results
    in it being returned immediately:

        >>> matrix = np.eye(3)
        >>> matrix
        array([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]])
        >>> is_PD(matrix)
        True
        >>> nearest_PD(matrix)
        array([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]])


    Using a real non-PD matrix results in it being transformed into an
    PD-matrix:

        >>> matrix = np.array([[1, 2], [3, 4]])
        >>> matrix
        array([[1, 2],
               [3, 4]])
        >>> is_PD(matrix)
        Traceback (most recent call last):
            ...
        ValueError: Input argument 'matrix' must be Hermitian!
        >>> mat_PD = nearest_PD(matrix)
        >>> mat_PD
        array([[ 1.31461828,  2.32186616],
               [ 2.32186616,  4.10085767]])
        >>> is_PD(mat_PD)
        True


    Using a complex non-PD matrix converts it into the nearest complex
    PD-matrix:

        >>> matrix = np.array([[4, 2+1j], [1+3j, 3]])
        >>> matrix
        array([[ 4.+0.j,  2.+1.j],
               [ 1.+3.j,  3.+0.j]])
        >>> mat_PD = nearest_PD(matrix)
        >>> mat_PD
        array([[ 4.0+0.j,  1.5-1.j],
               [ 1.5+1.j,  3.0+0.j]])
        >>> is_PD(mat_PD)
        True

    See also
    --------
    - :func:`~e13tools.math.is_PD`: Checks if `matrix` is positive-definite \
        or not.
    """

    # Make sure that matrix is a numpy array
    matrix = np.array(matrix)

    # Check if input is a matrix
    if(matrix.ndim != 2):
        raise ShapeError("Input argument 'matrix' must be two-dimensional!")
    else:
        rows, columns = matrix.shape

    # Check if matrix is a square
    if(rows != columns):
        raise ShapeError("Input argument 'matrix' has shape [%s, %s]. 'matrix'"
                         " must be a square matrix!" % (rows, columns))

    # Check if matrix is not already positive-definite
    try:
        is_PD(matrix)
    except ValueError:
        pass
    else:
        if is_PD(matrix):
            return(matrix)

    # Make sure that the matrix is Hermitian
    mat_H = (matrix+transposeC(matrix))/2

    # Perform singular value decomposition
    _, S, VH = svd(mat_H)

    # Compute the symmetric polar factor of mat_H
    spf = np.dot(transposeC(VH), np.dot(np.diag(S), VH))

    # Obtain the positive-definite matrix candidate
    mat_PD = (mat_H+spf)/2

    # Ensure that mat_PD is Hermitian
    mat_PD = (mat_PD+transposeC(mat_PD))/2

    # Check if mat_PD is in fact positive-definite
    if is_PD(mat_PD):
        return(mat_PD)

    # If it is not, change it very slightly to make it positive-definite
    In = np.eye(rows)
    k = 1
    spacing = np.spacing(norm(matrix))
    while not is_PD(mat_PD):
        min_eig_val = np.min(np.real(eigvals(mat_PD)))
        mat_PD += In*(-1*min_eig_val*pow(k, 2)+spacing)
        k += 1
    else:
        return(mat_PD)


def nPr(n, r, repeat=False):
    """
    For a given set S of `n` elements, returns the number of ordered
    arrangements ("permutations") of length `r` one can make with S.
    Returns zero if `r` > `n` and `repeat` is *False*.

    Parameters
    ----------
    n : int
        Number of elements in the set S.
    r : int
        Number of elements in the sub-set of set S.

    Optional
    --------
    repeat : bool. Default: False
        If *False*, each element in S can only be chosen once.
        If *True*, they can be chosen more than once.

    Returns
    -------
    n_perm : int
        Number of "permutations" that can be made with S.

    Examples
    --------
    >>> nPr(4, 2)
    12


    >>> nPr(4, 2, repeat=True)
    16


    >>> nPr(2, 4, repeat=True)
    16


    >>> nPr(2, 4)
    0

    See also
    --------
    - :func:`~e13tools.math.nCr`: Returns the number of unordered arrangements.

    """

    # Check if repeat is True or not and act accordingly
    if repeat:
        return(pow(n, r))
    elif(r == 0):
        return(1)
    elif(r > n):
        return(0)
    else:
        return(factorial(n)//factorial(n-r))


def rot90(array, axes=(0, 1), rot_axis='center', n_rot=1):
    """
    Rotates the given `array` by 90 degrees around the point `rot_axis` in the
    given `axes`. This function is different from NumPy's :func:`~numpy.rot90`
    function in that every column (2nd axis) defines a different dimension
    instead of every individual axis.

    Parameters
    ----------
    array : 2D array_like
        Array with shape [`n_pts`, `n_dim`] with `n_pts` the number of points
        and `n_dim` the number of dimensions. Requires: `n_dim` > 1.

    Optional
    --------
    axes : 1D array_like with 2 ints. Default: (0, 1)
        Array containing the axes defining the rotation plane. Rotation is from
        the first axis towards the second. Can be omitted if `rot_axis` has
        length `n_dim`.
    rot_axis : 1D array_like of length 2/`n_dim` or 'center'. Default: 'center'
        If 'center', the rotation axis is chosen in the center of the minimum
        and maximum values found in the given `axes`.
        If 1D array of length 2, the rotation axis is chosen around the given
        values in the given `axes`.
        If 1D array of length `n_dim`, the rotation axis is chosen around the
        first two non-zero values.
    n_rot : int. Default: 1
        Number of times to rotate `array` by 90 degrees.

    Returns
    -------
    array_rot : 2D :obj:`~numpy.ndarray` object
        Array with shape [`n_pts`, `n_dim`] that has been rotated by 90 degrees
        `n_rot` times.

    Examples
    --------
    Using an array with just two dimensions:

        >>> array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        >>> rot90(array)
        array([[ 1.  ,  0.75],
               [ 0.  ,  0.25],
               [ 0.25,  1.  ],
               [ 0.75,  0.  ]])


    Using the same array, but rotating it around a different point:

        >>> array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        >>> rot90(array, rot_axis=[0.2, 0.7])
        array([[ 0.9 ,  1.25],
               [-0.1 ,  0.75],
               [ 0.15,  1.5 ],
               [ 0.65,  0.5 ]])

    """

    # Make sure that array is a numpy array
    array = np.array(array)

    # Check if array is indeed two-dimensional and obtain the lengths
    if(array.ndim != 2):
        raise ShapeError("Input argument 'array' must be two-dimensional!")
    else:
        n_pts, n_dim = array.shape

    # Check axes
    axes = np.array(axes)
    if(axes.ndim == 1 and axes.shape[0] == 2 and (axes < n_dim).all()):
        pass
    else:
        raise InputError("Input argument 'axes' has invalid shape or values!")

    # Check what rot_axis is and act accordingly
    if(rot_axis == 'center'):
        rot_axis = np.zeros(2)
        rot_axis[0] =\
            abs(np.max(array[:, axes[0]])+np.min(array[:, axes[0]]))/2
        rot_axis[1] =\
            abs(np.max(array[:, axes[1]])+np.min(array[:, axes[1]]))/2
    elif(isinstance(rot_axis, str)):
        raise ValueError("Input argument 'rot_axis' can only have 'center' as"
                         " a string value!")
    else:
        rot_axis = np.array(rot_axis)
        if(rot_axis.ndim == 1 and rot_axis.shape[0] == 2):
            pass
        elif(rot_axis.ndim == 1 and rot_axis.shape[0] == n_dim):
            axes = []
            for i in range(n_dim):
                if(rot_axis[i] != 0):
                    axes.append(i)
                if(len(axes) == 2):
                    break
            else:
                raise ValueError("Input argument 'rot_axis' does not have two "
                                 "non-zero values!")
            rot_axis = rot_axis[axes]
        else:
            raise ShapeError("Input argument 'rot_axis' has invalid shape!")

    # Calculate the rotated matrix
    array_rot = array.copy()
    if(n_rot % 4 == 0):
        return(array_rot)
    elif(n_rot % 4 == 1):
        array_rot[:, axes[0]] = rot_axis[0]+rot_axis[1]-array[:, axes[1]]
        array_rot[:, axes[1]] = rot_axis[1]-rot_axis[0]+array[:, axes[0]]
    elif(n_rot % 4 == 2):
        array_rot[:, axes[0]] = 2*rot_axis[0]-array[:, axes[0]]
        array_rot[:, axes[1]] = 2*rot_axis[1]-array[:, axes[1]]
    elif(n_rot % 4 == 3):
        array_rot[:, axes[0]] = rot_axis[0]-rot_axis[1]+array[:, axes[1]]
        array_rot[:, axes[1]] = rot_axis[1]+rot_axis[0]-array[:, axes[0]]
    else:
        raise InputError("Input argument 'n_rot' is invalid!")

    # Return it
    return(array_rot)


def sort_2D(array, axis=-1, order=None):
    """
    Sorts a 2D `array` in a given `axis` in the specified `order`. This
    function is different from NumPy's :func:`~sort` function in that it sorts
    in a given axis rather than along it, and the order can be given as
    integers rather than field strings.

    Parameters
    ----------
    array : 2D array_like
        Input array that requires sorting.

    Optional
    --------
    axis : int. Default: -1
        Axis in which to sort the elements. Default is to sort all elements in
        the last axis. A negative value counts from the last to the first axis.
    order : int, 1D array_like of int or None. Default: None
        The order in which the vectors in the given `axis` need to be sorted.
        Negative values count from the last to the first vector.
        If *None*, all vectors in the given `axis` are sorted individually.

    Returns
    -------
    array_sort : 2D :obj:`~numpy.ndarray` object
        Input `array` with its `axis` sorted in the specified `order`.

    Examples
    --------
    Sorting the column elements of a given 2D array with no order specified:

        >>> array = np.array([[0, 5, 1], [7, 4, 9], [3, 13, 6], [0, 1, 8]])
        >>> array
        array([[ 0,  5,  1],
               [ 7,  4,  9],
               [ 3, 13,  6],
               [ 0,  1,  8]])
        >>> sort_2D(array)
        array([[ 0,  1,  1],
               [ 0,  4,  6],
               [ 3,  5,  8],
               [ 7, 13,  9]])


    Sorting the same array in only the first column:

        >>> sort_2D(array, order=(0))
        array([[ 0,  5,  1],
               [ 0,  1,  8],
               [ 3, 13,  6],
               [ 7,  4,  9]])


    Sorting all three columns in order:

        >>> sort_2D(array, order=(0, 1, 2))
        array([[ 0,  1,  8],
               [ 0,  5,  1],
               [ 3, 13,  6],
               [ 7,  4,  9]])


    Sorting all three columns in a different order:

        >>> sort_2D(array, order=(0, 2, 1))
        array([[ 0,  5,  1],
               [ 0,  1,  8],
               [ 3, 13,  6],
               [ 7,  4,  9]])

    """

    # Make sure that input array is a numpy array
    array = np.array(array)

    # Check if array is indeed 2D
    if(array.ndim != 2):
        raise ShapeError("Input argument 'array' must be two-dimensional!")
    else:
        # Obtain the number of vectors along the given axis
        n_vec = array.shape[axis]

    # Move the given axis to be the first axis
    try:
        array = np.moveaxis(array, axis, 0)
    except Exception as error:
            raise InputError("Input argument 'axis' is invalid (%s)!"
                             % (error))

    # If order is given, transform it into an array
    if order is not None:
        order = np.array(order, ndmin=1)

    # Check what order is given and act accordingly
    if order is None:
        array.sort(axis=-1)
    elif not(((-n_vec <= order)*(order < n_vec)).any()):
        raise ValueError("Input argument 'order' contains values that are "
                         "out of bounds!")
    else:
        for i in reversed(order):
            array = array[:, np.argsort(array[i], kind='mergesort')]

    # Return the resulting array back after transforming its axes back
    return(np.moveaxis(array, 0, axis))


def transposeC(array, axes=None):
    """
    Returns the (conjugate) transpose of the input `array`.

    Parameters
    ----------
    array : array_like
        Input array that needs to be transposed.

    Optional
    --------
    axes : 1D array_like of int or None. Default: None
        If *None*, reverse the dimensions.
        Else, permute the axes according to the values given.

    Returns
    -------
    array_t : :obj:`~numpy.ndarray` object
        Input `array` with its axes transposed.

    Examples
    --------
    Using an array with only real values returns its transposed variant:

        >>> array = np.array([[1, 2.5], [3.5, 5]])
        >>> array
        array([[ 1. ,  2.5],
               [ 3.5,  5. ]])
        >>> transposeC(array)
        array([[ 1. ,  3.5],
               [ 2.5,  5. ]])


    And using an array containing complex values returns its conjugate
    transposed:

        >>> array = np.array([[1, -2+4j], [7.5j, 0]])
        >>> array
        array([[ 1.+0.j , -2.+4.j ],
               [ 0.+7.5j,  0.+0.j ]])
        >>> transposeC(array)
        array([[ 1.-0.j ,  0.-7.5j],
               [-2.-4.j ,  0.-0.j ]])

    """

    # Take the transpose of the conjugate or the input array and return it
    return(np.transpose(np.conjugate(array), axes))
