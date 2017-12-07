# -*- coding: utf-8 -*-

"""
Math Core
=========
Provides a collection of functions that are core to Math and are imported
automatically.

"""


# %% IMPORTS
from __future__ import division, absolute_import, print_function

from math import factorial as _factorialm
import e13tools as e13
import numpy as np

__all__ = ['diff', 'is_PD', 'nCr', 'nearest_PD', 'nPr', 'transposeC']


# %% FUNCTIONS
def diff(array1, array2=None, flatten=True):
    """
    Calculates the pair-wise differences between inputs `array1` and `array2`
    in the outer dimension. If the biggest input is 2D, the output contains the
    1D pair-wise differences. If this input is 1D, the output contains the 0D
    pair-wise differences.

    Parameters
    ----------
    array1 : 0D, 1D or 2D array_like
        One of the inputs used to calculate the pair-wise differences.

    Optional
    --------
    array2 : 0D, 1D or 2D array_like. Default: None
        The other input used to calculate the pair-wise differences.
        If *None*, `array2` is equal to `array1`.
    flatten : bool. Default: True
        If `array2` is *None*, whether or not to calculate all pair-wise
        differences.
        If *True*, a flattened array containing all above-diagonal pair-wise
        differences is returned. This is useful if only off-diagonal terms are
        required and the sign is not important.
        If *False*, an array with all pair-wise differences is returned.

    Returns
    -------
    diff_array : 0D, 1D, 2D or 3D array_like
        Depending on the input parameters, an array with n dimensions
        containing the pair-wise differences between `array1` and `array2` in
        the outer dimension.

    Examples
    --------
    Using two matrices returns the pair-wise differences in vectors:

        >>> mat1 = np.array([[1, 2, 3], [4, 5, 6]])
        >>> mat2 = np.array([[4, 5, 6], [7, 8, 9]])
        >>> diff(mat1, mat2)
        array([[[-3., -3., -3.],
                [-6., -6., -6.]],
        <BLANKLINE>
               [[ 0.,  0.,  0.],
                [-3., -3., -3.]]])


    Only using a single matrix returns the pair-wise differences in vectors in
    that matrix (either flattened or not):

        >>> mat = np.array([[1, 2, 3], [4, 5, 6]])
        >>> diff(mat, flatten=True)
        array([[-3., -3., -3.]])
        >>> diff(mat, flatten=False)
        array([[[ 0.,  0.,  0.],
                [-3., -3., -3.]],
        <BLANKLINE>
               [[ 3.,  3.,  3.],
                [ 0.,  0.,  0.]]])


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
        # Make sure that input are numpy arrays
        array1 = np.array(array1)

        # Obtain the dimensionality of the input argument
        n_dim = array1.ndim

        # If n_dim is more than two, raise an Exception
        if(n_dim > 2):
            raise e13.ShapeError("Input must be a scalar, vector or matrix!")
        # If a matrix is provided, calculate the difference between all vectors
        elif(n_dim == 2):
            # Obtain the number of samples and values in the input matrix
            n_sam, n_val = array1.shape

            # If only the off-diagonal terms are requested
            if flatten is True:
                n_diff = n_sam*(n_sam-1)//2
                diff_array = np.zeros([n_diff, n_val])

                dist = 0
                for i in range(n_sam):
                    diff_array[dist:dist+n_sam-i-1] = array1[i]-array1[i+1:]
                    dist += n_sam-i-1

                return(diff_array)
            # If the full matrix is requested
            else:
                diff_array = np.zeros([n_sam, n_sam, n_val])

                for i in range(n_sam):
                    diff_array[i] = array1[i]-array1

                return(diff_array)
        # If a vector is provided, calculate the difference between all scalars
        elif(n_dim == 1):
            # Obtain the number of values in the input matrix
            n_val = array1.size

            # If only the off-diagonal terms are requested
            if flatten is True:
                n_diff = n_val*(n_val-1)//2
                diff_array = np.zeros([n_diff])

                dist = 0
                for i in range(n_val):
                    diff_array[dist:dist+n_val-i-1] = array1[i]-array1[i+1:]
                    dist += n_val-i-1

                return(diff_array)
            # If the full matrix is requested
            else:
                diff_array = np.zeros([n_val, n_val])

                for i in range(n_val):
                    diff_array[i] = array1[i]-array1

                return(diff_array)
        # If a scalar is provided, simply return the difference
        else:
            return(0)

    # If array2 is provided, both arrays are different
    else:
        # Make sure that input are numpy arrays
        array1 = np.array(array1)
        array2 = np.array(array2)

        # Obtain the dimensionality of the input arguments
        n_dim1 = array1.ndim
        n_dim2 = array2.ndim

        # If either n_dim is more than two, raise an Exception
        if(n_dim1 > 2 or n_dim2 > 2):
            raise e13.ShapeError("Input must be a scalar, vector or matrix!")
        # If both are matrices, calculate the difference between all vectors
        elif(n_dim1 == 2 and n_dim2 == 2):
            # Get number of samples and values in both matrices
            n_sam1, n_val1 = array1.shape
            n_sam2, n_val2 = array2.shape

            # Check if number of values are equal
            if(n_val1 != n_val2):
                raise e13.ShapeError("Input has incompatible number of values:"
                                     " %s != %s" % (n_val1, n_val2))

            diff_array = np.zeros([n_sam1, n_sam2, n_val1])

            for i in range(n_sam1):
                diff_array[i] = array1[i]-array2

            return(diff_array)
        # If both are vectors, calculate the difference between all scalars
        elif(n_dim1 == 1 and n_dim2 == 1):
            # Get number of values in both vectors
            n_val1 = array1.size
            n_val2 = array2.size

            diff_array = np.zeros([n_val1, n_val2])

            for i in range(n_val1):
                diff_array[i] = array1[i]-array2

            return(diff_array)
        # If a matrix and a vector are provided, return the difference in
        # vectors
        elif(n_dim1+n_dim2 == 3):
            # Obtain the length of the last dimension of the input arguments
            dim_l1 = array1.shape[-1]
            dim_l2 = array2.shape[-1]

            # Check if the last dimension has the same length
            if(dim_l1 != dim_l2):
                raise e13.ShapeError("Last dimension of inputs do not match: "
                                     "%s != %s" % (dim_l1, dim_l2))
            else:
                return(array1-array2)
        # If a matrix and a scalar are provided, raise an Exception
        elif(n_dim1+n_dim2 == 2):
            raise e13.ShapeError("Pair-wise difference cannot be calculated "
                                 "between a matrix and a vector!")
        # If a vector and a scalar or two scalars are provided, return the
        # difference in scalars
        else:
            return(array1-array2)


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
        ValueError: Input matrix must be Hermitian!


    Using a complex matrix that is positive-definite:

        >>> matrix = np.array([[4, 1.5+1j], [1.5-1j, 3]])
        >>> matrix
        array([[ 4.0+0.j,  1.5+1.j],
               [ 1.5-1.j,  3.0+0.j]])
        >>> is_PD(matrix)
        True

    """

    # Make sure that matrix is a numpy array
    matrix = np.array(matrix)

    # Check if input is a matrix
    if(matrix.ndim != 2):
        raise e13.ShapeError("Input must be two-dimensional!")
    else:
        rows, columns = matrix.shape

    # Check if matrix is a square
    if(rows != columns):
        raise e13.ShapeError("Input matrix has shape [%s, %s]. Input matrix "
                             "must be a square matrix!" % (rows, columns))

    # Check if matrix is Hermitian
    if not((transposeC(matrix) == matrix).all()):
        raise ValueError("Input matrix must be Hermitian!")

    # Try to use Cholesky on matrix. If it fails,
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return(False)
    else:
        return(True)


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
    - :func:`~e13tools.math.factorial`
    - :func:`~e13tools.math.nPr`: Returns the number of ordered arrangements.

    """

    # Check if repeat is True or not and act accordingly
    if repeat is True:
        return(_factorialm(n+r-1)//(_factorialm(r)*_factorialm(n-1)))
    elif r in (0, n):
        return(1)
    elif(r > n):
        return(0)
    else:
        return(_factorialm(n)//(_factorialm(r)*_factorialm(n-r)))


def nearest_PD(matrix):
    """
    Find the nearest positive-definite matrix to the input `matrix`.

    Parameters
    ----------
    matrix : 2D array_like
        Input matrix that requires its nearest positive-definite variant.

    Returns
    -------
    mat_PD : 2D array_like
        The nearest positive-definite matrix to the input `matrix`.

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
        ValueError: Input matrix must be Hermitian!
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

    """

    # Make sure that matrix is a numpy array
    matrix = np.array(matrix)

    # Check if input is a matrix
    if(matrix.ndim != 2):
        raise e13.ShapeError("Input must be two-dimensional!")
    else:
        rows, columns = matrix.shape

    # Check if matrix is a square
    if(rows != columns):
        raise e13.ShapeError("Input matrix has shape [%s, %s]. Input matrix "
                             "must be a square matrix!" % (rows, columns))

    # Check if matrix is not already positive-definite
    try:
        is_PD(matrix)
    except ValueError:
        pass
    else:
        if is_PD(matrix) is True:
            return(matrix)

    # Make sure that the matrix is Hermitian
    mat_H = (matrix+transposeC(matrix))/2

    # Perform singular value decomposition
    U, S, VH = np.linalg.svd(mat_H)

    # Compute the symmetric polar factor of mat_H
    spf = np.dot(transposeC(VH), np.dot(np.diag(S), VH))

    # Obtain the positive-definite matrix candidate
    mat_PD = (mat_H+spf)/2

    # Ensure that mat_PD is Hermitian
    mat_PD = (mat_PD+transposeC(mat_PD))/2

    # Check if mat_PD is in fact positive-definite
    if is_PD(mat_PD) is True:
        return(mat_PD)

    # If it is not, change it very slightly to make it positive-definite
    In = np.eye(rows)
    k = 1
    spacing = np.spacing(np.linalg.norm(matrix))
    while is_PD(mat_PD) is not True:
        min_eig_val = np.min(np.real(np.linalg.eigvals(mat_PD)))
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
    - :func:`~e13tools.math.factorial`
    - :func:`~e13tools.math.nCr`: Returns the number of unordered arrangements.

    """

    # Check if repeat is True or not and act accordingly
    if repeat is True:
        return(pow(n, r))
    elif(r == 0):
        return(1)
    elif(r > n):
        return(0)
    else:
        return(_factorialm(n)//_factorialm(n-r))


def transposeC(array):
    """
    Returns the (conjugate) transpose of the input `array`.

    Parameters
    ----------
    array : array_like
        Input array that needs to be transposed.

    Returns
    -------
    mat_t : array_like
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
    return(np.transpose(np.conjugate(array)))


# %% DOCTEST
if __name__ == '__main__':
    import doctest
    doctest.testmod()
