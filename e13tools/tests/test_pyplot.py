# -*- coding: utf-8 -*-

# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Built-in imports
import os
from os import path

# Package imports
import astropy.units as apu
from matplotlib import cm
import matplotlib.pyplot as plt
import pytest
from pytest_mpl.plugin import switch_backend

# e13Tools imports
from e13tools.core import InputError
from e13tools.pyplot import (apu2tex, center_spines, draw_textline, f2tex,
                             import_cmaps, q2tex)

# Save the path to this directory
dirpath = path.dirname(__file__)


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for apu2tex()-function
def test_apu2tex():
    assert apu2tex(apu.solMass) == r"\mathrm{M_{\odot}}"
    assert apu2tex(apu.solMass/apu.yr, unitfrac=False) ==\
        r"\mathrm{M_{\odot}\,yr^{-1}}"
    assert apu2tex(apu.solMass/apu.yr, unitfrac=True) ==\
        r"\mathrm{\frac{M_{\odot}}{yr}}"


# Pytest class for center_spines()-function
class Test_center_spines(object):
    # Test if default values work
    def test_default(self):
        with switch_backend('Agg'):
            fig = plt.figure()
            center_spines()
            plt.close(fig)

    # Test if setting the x and y tickers work
    def test_set_tickers(self):
        with switch_backend('Agg'):
            fig = plt.figure()
            plt.plot([-1, 1], [-1, 1])
            center_spines(set_xticker=1, set_yticker=1)
            plt.close(fig)


# Pytest class for draw_textline()-function
class Test_draw_textline(object):
    # Test if writing 'test' on the x-axis works, number 1
    def test_x_axis1(self):
        with switch_backend('Agg'):
            fig = plt.figure()
            draw_textline("test", x=-1, text_kwargs={'va': None})
            plt.close(fig)

    # Test if writing 'test' on the x-axis works, number 2
    def test_x_axis2(self):
        with switch_backend('Agg'):
            fig = plt.figure()
            draw_textline("test", x=2)
            plt.close(fig)

    # Test if writing 'test' on the y-axis works, number 1
    def test_y_axis1(self):
        with switch_backend('Agg'):
            fig = plt.figure()
            draw_textline("test", y=-1)
            plt.close(fig)

    # Test if writing 'test' on the y-axis works, number 2
    def test_y_axis2(self):
        with switch_backend('Agg'):
            fig = plt.figure()
            draw_textline("test", y=2)
            plt.close(fig)

    # Test if writing 'test' on the x-axis works, end-top pos
    def test_x_axis_end_top(self):
        with switch_backend('Agg'):
            fig = plt.figure()
            draw_textline("test", x=-1, pos="end top")
            plt.close(fig)

    # Test if writing 'test' on the x-axis works, start-bottom pos
    def test_x_axis_start_bottom(self):
        with switch_backend('Agg'):
            fig = plt.figure()
            draw_textline("test", x=-1, pos="start bottom")
            plt.close(fig)

    # Test if writing 'test' on the x-axis works, end-bottom pos
    def test_x_axis_end_bottom(self):
        with switch_backend('Agg'):
            fig = plt.figure()
            draw_textline("test", x=-1, pos="end bottom")
            plt.close(fig)

    # Test if writing 'test' on the x-axis y-axis fails
    def test_xy_axis(self):
        with switch_backend('Agg'):
            fig = plt.figure()
            with pytest.raises(InputError):
                draw_textline("test", x=-1, y=-1)
            plt.close(fig)

    # Test if writing 'test' on the x-axis fails for invalid pos
    def test_x_axis_invalid_pos(self):
        with switch_backend('Agg'):
            fig = plt.figure()
            with pytest.raises(ValueError):
                draw_textline("test", x=-1, pos="test")
            plt.close(fig)


# Pytest for f2tex()-function
def test_f2tex():
    assert f2tex(20.2935826592) == "20.29"
    assert f2tex(20.2935826592, sdigits=6) == "20.2936"
    assert f2tex(20.2935826592, power=1) == r"2.029\cdot 10^{1}"
    assert f2tex(1e6, nobase1=True) == "10^{6}"
    assert f2tex(1e6, nobase1=False) == r"1\cdot 10^{6}"
    assert f2tex(0) == "0"


# Pytest class for import_cmaps()-function
class Test_import_cmaps(object):
    # Test if providing a cmap file works
    def test_cmap_file(self):
        import_cmaps(path.join(dirpath, '../colormaps/cm_rainforest.txt'))

    # Test if all colormaps in e13tools/colormaps are loaded into MPL
    def test_MPL_cmaps(self):
        # Obtain path to directory with colormaps
        cmap_dir = path.abspath(path.join(dirpath, '../colormaps'))

        # Obtain list of all colormaps defined in e13Tools
        # As all colormaps have their own directories, save them instead
        cm_names = next(os.walk(cmap_dir))[1]

        # Add the reversed versions to the list as well
        cm_names.extend([cm_name+'_r' for cm_name in cm_names])

        # Obtain list of all colormaps registered in MPL
        cm_list = plt.colormaps()

        # Check if all names in cm_names are registered in MPL
        for cm_name in cm_names:
            assert hasattr(cm, cm_name)
            assert getattr(cm, cm_name) is plt.get_cmap(cm_name)
            assert cm_name in cm_list

    # Test if providing a non-existing directory raises an error
    def test_non_existing_dir(self):
        with pytest.raises(OSError):
            import_cmaps('./test')

    # Test if providing an invalid cmap file raises an error
    def test_invalid_cmap_file(self):
        with pytest.raises(OSError):
            import_cmaps(path.join(dirpath, 'data/test.txt'))

    # Test if providing an invalid cmap .npy-file raises an error
    def test_invalid_cmap_npy_file(self):
        with pytest.raises(InputError):
            import_cmaps(path.join(dirpath, 'data/cm_test2.npy'))

    # Test if providing a custom directory with invalid cmaps raises an error
    def test_invalid_cmap_dir(self):
        with pytest.raises(InputError):
            import_cmaps(path.join(dirpath, 'data'))


# Pytest for q2tex()-function
def test_q2tex():
    assert q2tex(20.2935826592) == "20.29"
    assert q2tex(20.2935826592*apu.solMass/apu.yr) ==\
        r"20.29\,\mathrm{M_{\odot}\,yr^{-1}}"
    assert q2tex(20.2935826592*apu.solMass/apu.yr, sdigits=6) ==\
        r"20.2936\,\mathrm{M_{\odot}\,yr^{-1}}"
    assert q2tex(20.2935826592*apu.solMass/apu.yr, power=1) ==\
        r"2.029\cdot 10^{1}\,\mathrm{M_{\odot}\,yr^{-1}}"
    assert q2tex(1e6*apu.solMass/apu.yr, nobase1=True) ==\
        r"10^{6}\,\mathrm{M_{\odot}\,yr^{-1}}"
    assert q2tex(1e6*apu.solMass/apu.yr, nobase1=False) ==\
        r"1\cdot 10^{6}\,\mathrm{M_{\odot}\,yr^{-1}}"
    assert q2tex(20.2935826592*apu.solMass/apu.yr, unitfrac=False) ==\
        r"20.29\,\mathrm{M_{\odot}\,yr^{-1}}"
    assert q2tex(20.2935826592*apu.solMass/apu.yr, unitfrac=True) ==\
        r"20.29\,\mathrm{\frac{M_{\odot}}{yr}}"
