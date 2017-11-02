# -*- coding: utf-8 -*-

"""
Math Core
===========
Provides a collection of functions that are core to Math and are imported
automatically.

"""


# %% IMPORTS
from __future__ import division, absolute_import, print_function

from e13tools import ShapeError
import numpy as np

__all__ = ['is_PD', 'nearest_SPD', 'transposeC']


# %% FUNCTIONS
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
    Using a matrix that is positive-definite (like the identity matrix):

    >>> matrix = np.eye(3)
    >>> matrix
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> is_PD(matrix)
    True


    Using a matrix that is not positive-definite (or symmetric):

    >>> matrix = np.array([[1, 2], [3, 4]])
    >>> matrix
    array([[1, 2],
           [3, 4]])
    >>> is_PD(matrix)
    False

    """

    # Check if matrix is Hermitian
    if((transposeC(matrix) != matrix).all()):
        raise ValueError("Input matrix must be Hermitian!")

    # Try to use Cholesky on matrix. If it fails,
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return(False)
    else:
        return(True)


def nearest_SPD(matrix):
    """
    Find the nearest symmetric positive-definite matrix to the input `matrix`.

    Parameters
    ----------
    matrix : 2D array_like
        Input array that requires its nearest symmetric positive-definite
        variant.

    Returns
    -------
    mat_SPD : 2D array_like
        The nearest symmetric positive-definite matrix to the input `matrix`.

    Examples
    --------
    Requesting the nearest SPD variant of a matrix that is already SPD results
    in it being returned immediately:

    >>> matrix = np.eye(3)
    >>> matrix
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> nearest_SPD(matrix)
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])


    Using a non-SPD matrix results in it being transformed into an SPD-matrix:

    >>> matrix = np.array([[1, 2], [3, 4]])
    >>> matrix
    array([[1, 2],
           [3, 4]])
    >>> nearest_SPD(matrix)
    array([[ 1.31461828,  2.32186616],
           [ 2.32186616,  4.10085767]])

    Notes
    -----
    This is a Python port of John D'Errico's *nearestSPD* code [1]_, which is a
    MATLAB implementation of Higham (1988) [2]_.

    According to Higham (1988), the nearest positive semi-definite matrix in
    the Frobenius norm to an arbitrary real matrix A is shown to be

        .. math:: \\frac{B+H}{2},

    with H being the symmetric polar factor of

        .. math:: B=\\frac{A+A^T}{2}.

    References
    ----------
    .. [1] \
        https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    .. [2] N.J. Higham, "Computing a Nearest Symmetric Positive Semidefinite
           Matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

    """

    # Check if input is a matrix
    if(np.shape(np.shape(matrix))[0] != 2):
        raise ShapeError("Input must be two-dimensional!")
    else:
        rows, columns = np.shape(matrix)

    # Check if matrix is a square
    if(rows != columns):
        raise ShapeError("Input matrix has shape [%s, %s]. Input matrix must "
                         "be a square matrix!" % (rows, columns))

    # Check if matrix is real
    if not(np.isreal(matrix).all()):
        raise ValueError("Input matrix must be real!")

    # Check if matrix is not already positive-definite
    if is_PD(matrix) is True:
        return(matrix)

    # Make sure that the matrix is symmetrized
    mat_sym = (matrix+transposeC(matrix))/2

    # Perform singular value decomposition
    U, S, VH = np.linalg.svd(mat_sym)

    # Compute the symmetric polar factor of mat_sym
    spf = np.dot(transposeC(VH), np.dot(np.diag(S), VH))

    # Obtain the positive-definite matrix candidate
    mat_PD = (mat_sym+spf)/2

    # Ensure that mat_PD is symmetric
    mat_SPD = (mat_PD+transposeC(mat_PD))/2

    # Check if mat_SPD is in fact positive-definite
    if is_PD(mat_SPD) is True:
        return(mat_SPD)

    # If it is not, change it very slightly to make it positive-definite
    I = np.eye(rows)
    k = 1
    spacing = np.spacing(np.linalg.norm(matrix))
    while is_PD(mat_SPD) is not True:
        min_eig_val = np.min(np.real(np.linalg.eigvals(mat_SPD)))
        mat_SPD += I*(-1*min_eig_val*pow(k, 2)+spacing)
        k += 1
    else:
        return(mat_SPD)


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

    >>> array = np.array([[1, 2], [3, 4]])
    >>> array
    array([[1, 2],
           [3, 4]])
    >>> transposeC(array)
    array([[1, 3],
           [2, 4]], dtype=int32)


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
