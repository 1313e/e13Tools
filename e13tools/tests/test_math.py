# -*- coding: utf-8 -*-

# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Package imports
import numpy as np
import pytest

# e13Tools imports
from e13tools.math import (diff, gcd, is_PD, lcm, nCr, nearest_PD, nPr, rot90,
                           sort2D, transposeC)


# %% PYTEST FUNCTIONS
class Test_diff(object):
    def test_matrices_row(self):
        mat1 = np.array([[1, 2, 3], [4, 5, 6]])
        mat2 = np.array([[4, 5, 6], [7, 8, 9]])
        assert np.all(diff(mat1, mat2) ==
                      np.array([[[-3, -3, -3], [-6, -6, -6]],
                                [[0, 0, 0], [-3, -3, -3]]]))

    def test_matrices_column(self):
        mat1 = np.array([[1, 2, 3], [4, 5, 6]])
        mat2 = np.array([[4, 5, 6], [7, 8, 9]])
        assert np.all(diff(mat1, mat2, axis=1) ==
                      np.array([[[-3, -3], [-4, -4], [-5, -5]],
                                [[-2, -2], [-3, -3], [-4, -4]],
                                [[-1, -1], [-2, -2], [-3, -3]]]))

    def test_single_matrix(self):
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        assert np.all(diff(mat) == np.array([[-3, -3, -3]]))
        assert np.all(diff(mat, flatten=True) == np.array([[-3, -3, -3]]))
        assert np.all(diff(mat, flatten=False) ==
                      np.array([[[0, 0, 0], [-3, -3, -3]],
                                [[3, 3, 3], [0, 0, 0]]]))

    def test_matrix_vector(self):
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        vec = np.array([7, 8, 9])
        assert np.all(diff(mat, vec) == np.array([[-6, -6, -6], [-3, -3, -3]]))

    def test_vectors(self):
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([4, 5, 6])
        assert np.all(diff(vec1, vec2) ==
                      np.array([[-3, -4, -5], [-2, -3, -4], [-1, -2, -3]]))


def test_gcd():
    assert gcd(18, 60, 72, 138) == 6


class Test_is_PD(object):
    def test_real_PD_matrix(self):
        matrix = np.eye(3)
        assert is_PD(matrix)

    def test_non_Hermitian_matrix(self):
        matrix = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError,
                           match="Input argument 'matrix' must be Hermitian!"):
            is_PD(matrix)

    def test_complex_PD_matrix(self):
        matrix = np.array([[4, 1.5+1j], [1.5-1j, 3]])
        assert is_PD(matrix)


def test_lcm():
    assert lcm(8, 9, 21) == 504


def test_nCr():
    assert nCr(4, 2) == 6
    assert nCr(4, 2, repeat=True) == 10
    assert nCr(2, 4, repeat=True) == 5
    assert nCr(2, 4) == 0


class Test_nearest_PD(object):
    def test_real_PD_matrix(self):
        matrix = np.eye(3)
        assert is_PD(matrix)
        assert np.all(nearest_PD(matrix) == matrix)

    def test_real_non_PD_matrix(self):
        matrix = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError,
                           match="Input argument 'matrix' must be Hermitian!"):
            is_PD(matrix)
        mat_PD = nearest_PD(matrix)
        assert is_PD(mat_PD)
        assert np.allclose(mat_PD, np.array([[1.31461828, 2.32186616],
                                             [2.32186616, 4.10085767]]))

    def test_complex_non_PD_matrix(self):
        matrix = np.array([[4, 2+1j], [1+3j, 3]])
        mat_PD = nearest_PD(matrix)
        assert is_PD(mat_PD)
        assert np.allclose(mat_PD, np.array([[4.0+0.j, 1.5-1.j],
                                             [1.5+1.j, 3.0+0.j]]))


def test_nPr():
    assert nPr(4, 2) == 12
    assert nPr(4, 2, repeat=True) == 16
    assert nPr(2, 4, repeat=True) == 16
    assert nPr(2, 4) == 0


class Test_rot90(object):
    def test_default(self):
        array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        assert np.allclose(rot90(array), np.array([[1, 0.75],
                                                   [0, 0.25],
                                                   [0.25, 1],
                                                   [0.75, 0]]))

    def test_rot_axis(self):
        array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        assert np.allclose(rot90(array, rot_axis=[0.2, 0.7]),
                           np.array([[0.9, 1.25], [-0.1, 0.75],
                                     [0.15, 1.5], [0.65, 0.5]]))


class Test_sort2D(object):
    def test_default(self):
        array = np.array([[0, 5, 1], [7, 4, 9], [3, 13, 6], [0, 1, 8]])
        assert np.allclose(sort2D(array), np.array([[0, 1, 1], [0, 4, 6],
                                                    [3, 5, 8], [7, 13, 9]]))

    def test_first_col(self):
        array = np.array([[0, 5, 1], [7, 4, 9], [3, 13, 6], [0, 1, 8]])
        assert np.allclose(sort2D(array, order=0),
                           np.array([[0, 5, 1], [0, 1, 8],
                                     [3, 13, 6], [7, 4, 9]]))

    def test_in_order(self):
        array = np.array([[0, 5, 1], [7, 4, 9], [3, 13, 6], [0, 1, 8]])
        assert np.allclose(sort2D(array, order=(0, 1, 2)),
                           np.array([[0, 1, 8], [0, 5, 1],
                                     [3, 13, 6], [7, 4, 9]]))

    def test_diff_order(self):
        array = np.array([[0, 5, 1], [7, 4, 9], [3, 13, 6], [0, 1, 8]])
        assert np.allclose(sort2D(array, order=(0, 2, 1)),
                           np.array([[0, 5, 1], [0, 1, 8],
                                     [3, 13, 6], [7, 4, 9]]))


class Test_transposeC(object):
    def test_real(self):
        array = np.array([[1, 2.5], [3.5, 5]])
        assert np.allclose(transposeC(array), np.array([[1, 3.5], [2.5, 5]]))

    def test_complex(self):
        array = np.array([[1, -2+4j], [7.5j, 0]])
        assert np.allclose(transposeC(array), np.array([[1-0j, 0-7.5j],
                                                        [-2-4j, 0-0j]]))
