# -*- coding: utf-8 -*-

"""
PyPlotTools
=======
Provides a collection of functions useful in various plotting routines.
Recommended usage::

    import e13tools.pyplottools as e13ppt

"""
# %% IMPORTS
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as apu
# import e13tools as e13


# %% FUNCTIONS
def apu2tex(unit, unitfrac=False):
    """
    Transform an :class:`~astropy.units.core.Unit` object into a (La)TeX string
    for usage in :mod:`~matplotlib.pyplot.figure`.

    Parameters
    ----------
    unit : :class:`~astropy.units.core.Unit` object
        Unit to be transformed.
    unitfrac : bool, optional. Default: False
        Whether or not to write `unit` as a LaTeX fraction.

    Returns
    -------
    out : string
        String containing `unit` written in (La)TeX string.

    Examples
    --------
    >>> import astropy.units as apu
    >>> e13ppt.apu2tex(apu.solMass)
    '\\mathrm{M_{\\odot}}'

    >>> import astropy.units as apu
    >>> e13ppt.apu2tex(apu.solMass/apu.yr, unitfrac=False)
    '\\mathrm{M_{\\odot}\\,yr^{-1}}'

    >>> import astropy.units as apu
    >>> e13ppt.apu2tex(apu.solMass/apu.yr, unitfrac=True)
    '\\mathrm{\\frac{M_{\\odot}}{yr}}'

    """

    if unitfrac is False:
        string = unit.to_string('latex_inline')
    else:
        string = unit.to_string('latex')

    # Remove '$' from the string
    return(string.replace("$", ""))


def center_spines(centerx=0, centery=0, set_xticker=False,
                  set_yticker=False, ax=None):
    """
    Centers the axis spines at <`centerx`, `centery`> on the axis `ax` in a
    :mod:`~matplotlib.pyplot.figure`. Centers the axis spines at the origin by
    default.

    Parameters
    ----------
    centerx : int or float; optional. Default: 0
        Centers x-axis at value `centerx`.
    centery : int or float; optional. Default: 0
        Centers y-axis at value `centery`.
    set_xticker : int, float or False; optional. Default: False
        If int or float, sets the x-axis ticker to `set_xticker`.

        If *False*, let :mod:`~matplotlib.pyplot.figure` instance decide.
    set_yticker : int, float or False; optional. Default: False
        If int or float, sets the y-axis ticker to `set_yticker`.

        If *False*, let :mod:`~matplotlib.pyplot.figure` instance decide.
    ax : :class:`~matplotlib.axes._subplots.AxesSubplot` object or None; optional. Default: None
        If :class:`~matplotlib.axes._subplots.AxesSubplot` object, centers the
        axis spines of specified :mod:`~matplotlib.pyplot.figure`.

        If *None*, centers the axis spines of current
        :mod:`~matplotlib.pyplot.figure`.

    """

    # If no AxesSubplot object is provided, make one
    if ax is None:
        ax = plt.gca()

    # Set the axis's spines to be centered at the given point
    # (Setting all 4 spines so that the tick marks go in both directions)
    ax.spines['left'].set_position(('data', centerx))
    ax.spines['bottom'].set_position(('data', centery))
    ax.spines['right'].set_position(('data', centerx))
    ax.spines['top'].set_position(('data', centery))

    # Hide the line (but not ticks) for "extra" spines
    for side in ['right', 'top']:
        ax.spines[side].set_color('none')

    # On both the x and y axes...
    for axis, center in zip([ax.xaxis, ax.yaxis], [centerx, centery]):
        # STILL HAVE TO FIX THAT THE TICKLABELS ARE ALWAYS HIDDEN
        # Hide the ticklabels at <centerx, centery>
        formatter = CenteredFormatter()
        formatter.center = center
        axis.set_major_formatter(formatter)

    # Add origin offset ticklabel if <centerx=0, centery=0> using annotation
    if(centerx == 0 and centery == 0):
        xlabel, ylabel = map(formatter.format_data, [centerx, centery])
        ax.annotate("0", (centerx, centery), xytext=(-4, -4),
                    textcoords='offset points', ha='right', va='top')

    # Set x-axis ticker
    if set_xticker is not False:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(set_xticker))

    # Set y-axis ticker
    if set_yticker is not False:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(set_yticker))


class CenteredFormatter(mpl.ticker.ScalarFormatter):
    """
    Acts exactly like the default Scalar Formatter, but yields an empty
    label for ticks at "center".

    """

    center = 0

    def __call__(self, value, pos=None):
        if(value == self.center):
            print(self.center)
            return("")
        else:
            return(mpl.ticker.ScalarFormatter.__call__(self, value, pos))


def draw_textline(text, x=None, y=None, pos='start', ax=None):
    """
    Draws a dotted line on the axis `ax` in a
    :mod:`~matplotlib.pyplot.figure` instance and prints `text`
    on top.

    Parameters
    ----------
    text : string
        Text to be printed on the line.
    x : scalar or None
        If scalar, text/line x-coordinate.

        If *None*, line covers complete x-axis.
    y : scalar or None
        If scalar, text/line y-coordinate.

        If *None*, line covers complete y-axis.
    pos : 'start' or 'end'; optional. Default: 'start'
        If 'start', prints the text at the start of the drawn line.

        If 'end', prints the text at the end of the drawn line.
    ax : :class:`~matplotlib.axes._subplots.AxesSubplot` object or None; optional. Default: None
        If :class:`~matplotlib.axes._subplots.AxesSubplot` object, draws line
        in specified :mod:`~matplotlib.pyplot.figure`.

        If *None*, draws line in current :mod:`~matplotlib.pyplot.figure`.

    """

    # If no AxesSubplot object is provided, make one
    if ax is None:
        ax = plt.gca()

    if x is None and y is not None:
        # Draw a line
        ax.plot(ax.set_xlim(), [y, y], 'k:')

        if(pos == 'start'):
            ax.text(ax.set_xlim()[0], y, text, fontsize=14, color='k',
                    horizontalalignment='left', verticalalignment='bottom')
        elif(pos == 'end'):
            ax.text(ax.set_xlim()[1], y, text, fontsize=14, color='k',
                    horizontalalignment='right', verticalalignment='bottom')
        else:
            raise ValueError('ERROR: Unknown text positioning!')

        # Adjust axes to include text in plot
        ax_ysize = abs(ax.set_ylim()[1]-ax.set_ylim()[0])

        # Adjust axes if line is located on the bottom
        if(ax.set_ylim()[0] <= y and ax.set_ylim()[0] >= y-0.1*ax_ysize):
            ax.set_ylim(y-0.1*ax_ysize, ax.set_ylim()[1])
        elif(ax.set_ylim()[0] <= y and ax.set_ylim()[0] >= y+0.1*ax_ysize):
            ax.set_ylim(y+0.1*ax_ysize, ax.set_ylim()[1])

        # Adjust axes if line is located on the top
        if(ax.set_ylim()[1] >= y and ax.set_ylim()[1] <= y+0.1*ax_ysize):
            ax.set_ylim(ax.set_ylim()[0], y+0.1*ax_ysize)
        elif(ax.set_ylim()[1] >= y and ax.set_ylim()[1] <= y-0.1*ax_ysize):
            ax.set_ylim(ax.set_ylim()[0], y-0.1*ax_ysize)

    elif y is None and x is not None:
        ax.plot([x, x], ax.set_ylim(), 'k:')

        if(pos == 'start'):
            ax.text(x, ax.set_ylim()[0], text, fontsize=14, color='k',
                    rotation=90, horizontalalignment='right',
                    verticalalignment='bottom')
        elif(pos == 'end'):
            ax.text(x, ax.set_ylim()[1], text, fontsize=14, color='k',
                    rotation=90, horizontalalignment='right',
                    verticalalignment='top')
        else:
            raise ValueError('ERROR: Unknown text positioning!')

        # Adjust axes to include text in plot
        ax_xsize = abs(ax.set_xlim()[1]-ax.set_xlim()[0])

        # Adjust axes if line is located on the left
        if(ax.set_xlim()[0] <= x and ax.set_xlim()[0] >= x-0.1*ax_xsize):
            ax.set_xlim(x-0.1*ax_xsize, ax.set_xlim()[1])
        elif(ax.set_xlim()[0] <= x and ax.set_xlim()[0] >= x+0.1*ax_xsize):
            ax.set_xlim(x+0.1*ax_xsize, ax.set_xlim()[1])

        # Adjust axes for if line is located on the right
        if(ax.set_xlim()[1] >= x and ax.set_xlim()[1] <= x+0.1*ax_xsize):
            ax.set_xlim(ax.set_xlim()[0], x+0.1*ax_xsize)
        elif(ax.set_xlim()[1] >= x and ax.set_xlim()[1] <= x-0.1*ax_xsize):
            ax.set_xlim(ax.set_xlim()[0], x-0.1*ax_xsize)
    else:
        raise ValueError('ERROR: No single line axis was given!')


def f2tex(value, sdigits=4, power=3, nobase1=True):
    """
    Transform a value into a (La)TeX string for usage in
    :mod:`~matplotlib.pyplot.figure`.

    Parameters
    ----------
    value : int or float
        Value to be transformed.
    sdigits : int; optional. Default: 4
        Maximum amount of significant digits `value` is returned with.
    power : int; optional. Default: 3
        Minimum log10(`value`) required before `value` is written in
        scientific form.
    nobase1 : bool; optional. Default: True
        Whether or not to include `base` in scientific form if `base=1`.

    Returns
    -------
    out : string
        String containing `value` written in (La)TeX string.

    Examples
    --------
    >>> e13ppt.f2tex(20.2935826592)
    '20.29'

    >>> e13ppt.f2tex(20.2935826592, sdigits=6)
    '20.2936'

    >>> e13ppt.f2tex(20.2935826592, power=1)
    '2.029\\cdot 10^{1}'

    >>> e13ppt.f2tex(1e6, nobase1=True)
    '10^{6}'

    >>> e13ppt.f2tex(1e6, nobase1=False)
    '1\\cdot 10^{6}'

    """

    n = int(np.floor(np.log10(value)))

    if(abs(n) < power):
        string = r"%.{}g".format(sdigits) % (value)
    else:
        base = value/pow(10, n)
        if(base == 1 and nobase1 is True):
            string = r"10^{%i}" % (n)
        else:
            string = r"%.{}g\cdot 10^{{%i}}".format(sdigits) % (base, n)
    return(string)


def q2tex(quantity, sdigits=4, power=3, nobase1=True, unitfrac=False):
    """
    Combination of :func:`~e13tools.e13pyplottools.f2tex` and
    :func:`~e13tools.e13pyplottools.apu2tex`.

    Transform a quantity into a (La)TeX string for usage in
    :mod:`~matplotlib.pyplot.figure`.

    Parameters
    ----------
    quantity : int, float or :class:`~astropy.units.quantity.Quantity` object
        Quantity to be transformed.
    sdigits : int; optional. Default: 4
        Maximum amount of significant digits `value` is returned with.
    power : int; optional. Default: 3
        Minimum log10(`value`) required before `value` is written in
        scientific form.
    nobase1 : bool; optional. Default: True
        Whether or not to include `base` in scientific form if `base=1`.
    unitfrac : bool, optional. Default: False
        Whether or not to write `unit` as a LaTeX fraction.

    Returns
    -------
    out : string
        String containing `quantity` written in (La)TeX string.

    Examples
    --------
    >>> import astropy.units as apu
    >>> e13ppt.q2tex(20.2935826592*apu.solMass/apu.yr)
    '20.29\\ \\mathrm{M_{\\odot}\\,yr^{-1}}'

    >>> import astropy.units as apu
    >>> e13ppt.q2tex(20.2935826592*apu.solMass/apu.yr, sdigits=6)
    '20.2936\\ \\mathrm{M_{\\odot}\\,yr^{-1}}'

    >>> import astropy.units as apu
    >>> e13ppt.q2tex(20.2935826592*apu.solMass/apu.yr, power=1)
    '2.029\\ \\cdot 10^{1}\\mathrm{M_{\\odot}\\,yr^{-1}}'

    >>> import astropy.units as apu
    >>> e13ppt.q2tex(1e6*apu.solMass/apu.yr, nobase1=True)
    '10^{6}\\ \\mathrm{M_{\\odot}\\,yr^{-1}}'

    >>> import astropy.units as apu
    >>> e13ppt.q2tex(1e6*apu.solMass/apu.yr, nobase1=False)
    '1\\cdot 10^{6}\\ \\mathrm{M_{\\odot}\\,yr^{-1}}'

    >>> import astropy.units as apu
    >>> e13ppt.q2tex(20.2935826592*apu.solMass/apu.yr, unitfrac=False)
    '20.29\\ \\mathrm{M_{\\odot}\\,yr^{-1}}'

    >>> import astropy.units as apu
    >>> e13ppt.q2tex(20.2935826592*apu.solMass/apu.yr, unitfrac=True)
    '20.29\\ \\mathrm{\\frac{M_{\\odot}}{yr}}'

    """

    # Check if quantity has a unit
    if type(quantity) is apu.quantity.Quantity:
        value = quantity.value
        unit = quantity.unit
    else:
        value = quantity
        unit = 0

    # Value handling
    string = f2tex(value, sdigits, power, nobase1)

    # Unit handling
    if(unit):
        unit_string = apu2tex(unit, unitfrac)
        string = ''.join([string, '\\ ', unit_string])

    return(string)


# %% Main program if run directly
if(__name__ == '__main__'):
    A = 1e6*apu.solMass/apu.yr
    x = np.linspace(-150, 150, 1001)
    y = np.arctan(x)/(np.pi)
    plt.rc('text', usetex=True)
    center_spines(centerx=0.0, centery=0.0, set_yticker=0.2)
    plt.title(r"$A=%s$, $A=%s$" % (q2tex(A, unitfrac=True), q2tex(A)))
    plt.plot(x, y, label=r"$y = \frac{arctan(x)}{\pi}$")
    plt.axis([-150, 150, 1.5, -1.5])
    draw_textline('hoi', y=-1)
    print(plt.ylim())
    plt.legend(loc='upper left', fontsize=20)
    plt.show()
