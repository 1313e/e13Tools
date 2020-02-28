# -*- coding: utf-8 -*-

# %% IMPORTS
# Package imports
import numpy as np
import pytest

# e13Tools imports
from e13tools import InputError, ShapeError
from e13tools.math import (diff, gcd, is_PD, lcm, nCr, nearest_PD, nPr, rot90,
                           sort2D, transposeC)


# %% PYTEST FUNCTIONS
# Pytest class for the diff()-function
class Test_diff(object):
    # Test row difference between two matrices
    def test_matrices_row(self):
        mat1 = np.array([[1, 2, 3], [4, 5, 6]])
        mat2 = np.array([[4, 5, 6], [7, 8, 9]])
        assert np.allclose(diff(mat1, mat2),
                           np.array([[[-3, -3, -3], [-6, -6, -6]],
                                     [[0, 0, 0], [-3, -3, -3]]]))

    # Test column difference between two matrices
    def test_matrices_column(self):
        mat1 = np.array([[1, 2, 3], [4, 5, 6]])
        mat2 = np.array([[4, 5, 6], [7, 8, 9]])
        assert np.allclose(diff(mat1, mat2, axis=1),
                           np.array([[[-3, -3], [-4, -4], [-5, -5]],
                                     [[-2, -2], [-3, -3], [-4, -4]],
                                     [[-1, -1], [-2, -2], [-3, -3]]]))

    # Test difference of matrix with itself
    def test_single_matrix(self):
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        assert np.allclose(diff(mat), np.array([[-3, -3, -3]]))
        assert np.allclose(diff(mat, flatten=True), np.array([[-3, -3, -3]]))
        assert np.allclose(diff(mat, flatten=False),
                           np.array([[[0, 0, 0], [-3, -3, -3]],
                                     [[3, 3, 3], [0, 0, 0]]]))

    # Test difference between matrix and vector
    def test_matrix_vector(self):
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        vec = np.array([7, 8, 9])
        assert np.allclose(diff(mat, vec), np.array([[-6, -6, -6],
                                                     [-3, -3, -3]]))
        assert np.allclose(diff(vec, mat), np.array([[6, 6, 6],
                                                     [3, 3, 3]]))

    # Test difference between two vectors
    def test_vectors(self):
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([4, 5, 6])
        assert np.allclose(diff(vec1, vec2), np.array([[-3, -4, -5],
                                                       [-2, -3, -4],
                                                       [-1, -2, -3]]))

    # Test difference of vector with itself
    def test_single_vector(self):
        vec = np.array([7, 8, 9])
        assert np.allclose(diff(vec), [-1, -2, -1])

    # Test difference bwteen two scalars
    def test_scalars(self):
        assert diff(2, 1) == 1

    # Test difference of scalar with itself
    def test_single_scalar(self):
        assert diff(1) == 0

    # Test if invalid axis raises an error using a single vector
    def test_single_invalid_axis(self):
        vec = np.array([7, 8, 9])
        with pytest.raises(InputError):
            diff(vec, axis=1)

    # Test if invalid axis raises an error using two vectors
    def test_double_invalid_axis(self):
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([4, 5, 6])
        with pytest.raises(InputError):
            diff(vec1, vec2, axis=1)

    # Test if using matrices with different axes lengths raises an error
    def test_two_diff_axes(self):
        mat1 = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])
        mat2 = np.array([[4, 5, 6], [7, 8, 9]])
        with pytest.raises(ShapeError):
            diff(mat1, mat2)

    # Test if using matrix and vector with invalid axis raises an error
    def test_matrix_vector_invalid_axis(self):
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        vec = np.array([7, 8, 9])
        with pytest.raises(InputError):
            diff(mat, vec, axis=2)

    # Test if using matrix and vector with different axes lengths raises error
    def test_matrix_vector_diff_axes(self):
        mat = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])
        vec = np.array([7, 8, 9])
        with pytest.raises(ShapeError):
            diff(mat, vec)


# Do default test for gcd()-function
def test_gcd():
    assert gcd(18, 60, 72, 138) == 6


# Pytest class for the is_PD()-function
class Test_is_PD(object):
    # Test if real PD matrix returns True
    def test_real_PD_matrix(self):
        mat = np.eye(3)
        assert is_PD(mat)

    # Test if real non-PD matrix returns False
    def test_real_non_PD_matrix(self):
        mat = np.array([[1, 2.5], [2.5, 4]])
        assert not is_PD(mat)

    # Test if complex PD matrix returns True
    def test_complex_PD_matrix(self):
        mat = np.array([[4, 1.5+1j], [1.5-1j, 3]])
        assert is_PD(mat)

    # Test if using vector raises an error
    def test_vector(self):
        vec = np.array([7, 8, 9])
        with pytest.raises(ShapeError):
            is_PD(vec)

    # Test if using a non-square matrix raises an error
    def test_non_square_matrix(self):
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ShapeError):
            is_PD(mat)

    # Test if non-Hermitian matrix raises an error
    def test_non_Hermitian_matrix(self):
        mat = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            is_PD(mat)


# Do default test for lcm()-function
def test_lcm():
    assert lcm(8, 9, 21) == 504


# Pytest class for nCr()-function
class Test_nCr(object):
    # Test for repeat = False
    def test_no_repeat(self):
        assert nCr(4, 0) == 1
        assert nCr(4, 1) == 4
        assert nCr(4, 2) == 6
        assert nCr(4, 3) == 4
        assert nCr(4, 4) == 1
        assert nCr(4, 5) == 0

    # Test for repeat = True
    def test_with_repeat(self):
        assert nCr(4, 0, repeat=True) == 1
        assert nCr(4, 1, repeat=True) == 4
        assert nCr(4, 2, repeat=True) == 10
        assert nCr(4, 3, repeat=True) == 20
        assert nCr(4, 4, repeat=True) == 35
        assert nCr(4, 5, repeat=True) == 56


# Pytest class for nearest_PD()-function
class Test_nearest_PD(object):
    # Test if using a real PD matrix returns the matrix
    def test_real_PD_matrix(self):
        mat = np.eye(3)
        assert is_PD(mat)
        assert np.allclose(nearest_PD(mat), mat)

    # Test if using a real non-PD matrix converts it into a PD matrix
    def test_real_non_PD_matrix(self):
        mat = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            is_PD(mat)
        mat_PD = nearest_PD(mat)
        assert is_PD(mat_PD)
        assert np.allclose(mat_PD, np.array([[1.31461828, 2.32186616],
                                             [2.32186616, 4.10085767]]))

    # Test if using a complex non-PD matrix converts it into a PD matrix
    def test_complex_non_PD_matrix(self):
        mat = np.array([[4, 2+1j], [1+3j, 3]])
        mat_PD = nearest_PD(mat)
        assert is_PD(mat_PD)
        assert np.allclose(mat_PD, np.array([[4.0+0.j, 1.5-1.j],
                                             [1.5+1.j, 3.0+0.j]]))

    # Test if using vector raises an error
    def test_vector(self):
        vec = np.array([7, 8, 9])
        with pytest.raises(ShapeError):
            nearest_PD(vec)

    # Test if using a non-square matrix raises an error
    def test_non_square_matrix(self):
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ShapeError):
            nearest_PD(mat)


# Pytest class for nPr()-function
class Test_nPr(object):
    # Test for repeat = False
    def test_no_repeat(self):
        assert nPr(4, 0) == 1
        assert nPr(4, 1) == 4
        assert nPr(4, 2) == 12
        assert nPr(4, 3) == 24
        assert nPr(4, 4) == 24
        assert nPr(4, 5) == 0

    # Test for repeat = True
    def test_with_repeat(self):
        assert nPr(4, 0, repeat=True) == 1
        assert nPr(4, 1, repeat=True) == 4
        assert nPr(4, 2, repeat=True) == 16
        assert nPr(4, 3, repeat=True) == 64
        assert nPr(4, 4, repeat=True) == 256
        assert nPr(4, 5, repeat=True) == 1024


# Pytest class for rot90()-function
class Test_rot90(object):
    # Test if rotating with default values returns correct array
    def test_default(self):
        array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        assert np.allclose(rot90(array), np.array([[1, 0.75],
                                                   [0, 0.25],
                                                   [0.25, 1],
                                                   [0.75, 0]]))

    # Test if rotating 0 times with default values returns correct array
    def test_default_0(self):
        array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        assert np.allclose(rot90(array, n_rot=0), np.array([[0.75, 0],
                                                            [0.25, 1],
                                                            [1, 0.75],
                                                            [0, 0.25]]))

    # Test if rotating 1 time with default values returns correct array
    def test_default_1(self):
        array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        assert np.allclose(rot90(array, n_rot=1), np.array([[1, 0.75],
                                                            [0, 0.25],
                                                            [0.25, 1],
                                                            [0.75, 0]]))

    # Test if rotating 2 times with default values returns correct array
    def test_default_2(self):
        array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        assert np.allclose(rot90(array, n_rot=2), np.array([[0.25, 1],
                                                            [0.75, 0],
                                                            [0, 0.25],
                                                            [1, 0.75]]))

    # Test if rotating 3 times with default values returns correct array
    def test_default_3(self):
        array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        assert np.allclose(rot90(array, n_rot=3), np.array([[0, 0.25],
                                                            [1, 0.75],
                                                            [0.75, 0],
                                                            [0.25, 1]]))

    # Test if changing the 2D rotation axis returns correct array
    def test_2D_rot_axis(self):
        array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        assert np.allclose(rot90(array, rot_axis=[0.2, 0.7]),
                           np.array([[0.9, 1.25], [-0.1, 0.75],
                                     [0.15, 1.5], [0.65, 0.5]]))

    # Test if changing the 3D rotation axis returns correct array
    def test_3D_rot_axis(self):
        array = np.array([[0.75, 0, 0], [0.25, 1, 0], [1, 0.75, 0]])
        assert np.allclose(rot90(array, rot_axis=[0.2, 0.7, 0]),
                           np.array([[0.9, 1.25, 0], [-0.1, 0.75, 0],
                                     [0.15, 1.5, 0]]))

    # Test if using vector raises an error
    def test_vector(self):
        vec = np.array([7, 8, 9])
        with pytest.raises(ShapeError):
            rot90(vec)

    # Test if using three axes for rotation raises an error
    def test_3_rot_axes(self):
        array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        with pytest.raises(InputError):
            rot90(array, axes=(0, 1, 2))

    # Test if using an invalid rotation axis string raises an error
    def test_invalid_rot_axis_str(self):
        array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        with pytest.raises(ValueError):
            rot90(array, rot_axis='test')

    # Test if changing the 3D rotation axis incorrectly raises an error
    def test_invalid_3D_rot_axis(self):
        array = np.array([[0.75, 0, 0], [0.25, 1, 0], [1, 0.75, 0]])
        with pytest.raises(ValueError):
            rot90(array, rot_axis=[0.2, 0, 0])

    # Test if changing using incorrect rotation axis raises an error
    def test_4D_rot_axis(self):
        array = np.array([[0.75, 0, 0], [0.25, 1, 0], [1, 0.75, 0]])
        with pytest.raises(ShapeError):
            rot90(array, rot_axis=[0.2, 0.7, 0, 0])

    # Test if rotating 4.5 times with default values raises an error
    def test_default_4_5(self):
        array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        with pytest.raises(InputError):
            rot90(array, n_rot=4.5)


# Pytest class for sort2D()-function
class Test_sort2D(object):
    # Test with default values
    def test_default(self):
        array = np.array([[0, 5, 1], [7, 4, 9], [3, 13, 6], [0, 1, 8]])
        assert np.allclose(sort2D(array), np.array([[0, 1, 1], [0, 4, 6],
                                                    [3, 5, 8], [7, 13, 9]]))

    # Test if sorting on first column works
    def test_first_col(self):
        array = np.array([[0, 5, 1], [7, 4, 9], [3, 13, 6], [0, 1, 8]])
        assert np.allclose(sort2D(array, order=0),
                           np.array([[0, 5, 1], [0, 1, 8],
                                     [3, 13, 6], [7, 4, 9]]))

    # Test if sorting on first row works
    def test_first_row(self):
        array = np.array([[0, 5, 1], [7, 4, 9], [3, 13, 6], [0, 1, 8]])
        assert np.allclose(sort2D(array, axis=0, order=0),
                           np.array([[0, 1, 5], [7, 9, 4],
                                     [3, 6, 13], [0, 8, 1]]))

    # Test if sorting in order works
    def test_in_order(self):
        array = np.array([[0, 5, 1], [7, 4, 9], [3, 13, 6], [0, 1, 8]])
        assert np.allclose(sort2D(array, order=(0, 1, 2)),
                           np.array([[0, 1, 8], [0, 5, 1],
                                     [3, 13, 6], [7, 4, 9]]))

    # Test if sorting in different order works
    def test_diff_order(self):
        array = np.array([[0, 5, 1], [7, 4, 9], [3, 13, 6], [0, 1, 8]])
        assert np.allclose(sort2D(array, order=(0, 2, 1)),
                           np.array([[0, 5, 1], [0, 1, 8],
                                     [3, 13, 6], [7, 4, 9]]))

    # Test if using vector raises an error
    def test_vector(self):
        vec = np.array([7, 8, 9])
        with pytest.raises(ShapeError):
            sort2D(vec)

    # Test if using invalid axis raises an error
    def test_invalid_axis(self):
        array = np.array([[0, 5, 1], [7, 4, 9], [3, 13, 6], [0, 1, 8]])
        with pytest.raises(InputError):
            sort2D(array, axis=3)

    # Test if sorting in invalid order raises an error
    def test_invalid_order(self):
        array = np.array([[0, 5, 1], [7, 4, 9], [3, 13, 6], [0, 1, 8]])
        with pytest.raises(ValueError):
            sort2D(array, order=(0, 3, 1))


# Pytest class for transposeC()-function
class Test_transposeC(object):
    # Test if transposing a real array returns the correct transpose
    def test_real(self):
        array = np.array([[1, 2.5], [3.5, 5]])
        assert np.allclose(transposeC(array), np.array([[1, 3.5], [2.5, 5]]))

    # Test if transposing a complex array returns the correct transpose
    def test_complex(self):
        array = np.array([[1, -2+4j], [7.5j, 0]])
        assert np.allclose(transposeC(array), np.array([[1-0j, 0-7.5j],
                                                        [-2-4j, 0-0j]]))
