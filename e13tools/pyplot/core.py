# -*- coding: utf-8 -*-

"""
PyPlot Core
===========
Provides a collection of functions that are core to **PyPlot** and are imported
automatically.

Available functions
-------------------
:func:`~apu2tex`
    Transform a :obj:`~astropy.units.core.Unit` object into a (La)TeX string
    for usage in a :obj:`~matplotlib.figure.Figure` instance.

:func:`~center_spines`
    Centers the axis spines at <`centerx`, `centery`> on the axis `ax` in a
    :obj:`~matplotlib.figure.Figure` instance. Centers the axis spines at the
    origin by default.

:func:`~draw_textline`
    Draws a line on the axis `ax` in a :obj:`~matplotlib.figure.Figure`
    instance instance and prints `text` on top.

:func:`~f2tex`
    Transform a value into a (La)TeX string for usage in a
    :obj:`~matplotlib.figure.Figure` instance.

:func:`~q2tex`
    Combination of :func:`~e13tools.pyplot.f2tex` and
    :func:`~e13tools.pyplot.apu2tex`.
    Transform a :obj:`~astropy.units.quantity.Quantity` object into a (La)TeX
    string for usage in a :obj:`~matplotlib.figure.Figure` instance.

"""


# %% IMPORTS
from __future__ import absolute_import, division, print_function

from e13tools import InputError
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
try:
    import astropy.units as apu
    import_astropy = 1
except ImportError:
    import_astropy = 0

__all__ = ['apu2tex', 'center_spines', 'draw_textline', 'f2tex', 'q2tex',
           'suplabel']


# %% FUNCTIONS
def apu2tex(unit, unitfrac=False):
    """
    Transform a :obj:`~astropy.units.core.Unit` object into a (La)TeX string
    for usage in a :obj:`~matplotlib.figure.Figure` instance.

    Parameters
    ----------
    unit : :obj:`~astropy.units.core.Unit` object
        Unit to be transformed.

    Optional
    --------
    unitfrac : bool. Default: False
        Whether or not to write `unit` as a LaTeX fraction.

    Returns
    -------
    out : string
        String containing `unit` written in (La)TeX string.

    Examples
    --------
    >>> import astropy.units as apu
    >>> apu2tex(apu.solMass)
    '\\\\mathrm{M_{\\\\odot}}'

    >>> import astropy.units as apu
    >>> apu2tex(apu.solMass/apu.yr, unitfrac=False)
    '\\\\mathrm{M_{\\\\odot}\\\\,yr^{-1}}'

    >>> import astropy.units as apu
    >>> apu2tex(apu.solMass/apu.yr, unitfrac=True)
    '\\\\mathrm{\\\\frac{M_{\\\\odot}}{yr}}'

    """

    if import_astropy:
        if not unitfrac:
            string = unit.to_string('latex_inline')
        else:
            string = unit.to_string('latex')

        # Remove '$' from the string and make output a string (py2.7)
        return(str(string.replace("$", "")))
    else:
        raise ImportError("This function requires AstroPy!")


def center_spines(centerx=0, centery=0, set_xticker=False, set_yticker=False,
                  ax=None):
    """
    Centers the axis spines at <`centerx`, `centery`> on the axis `ax` in a
    :obj:`~matplotlib.figure.Figure` instance. Centers the axis spines at the
    origin by default.

    Optional
    --------
    centerx : int or float. Default: 0
        Centers x-axis at value `centerx`.
    centery : int or float. Default: 0
        Centers y-axis at value `centery`.
    set_xticker : int, float or False. Default: False
        If int or float, sets the x-axis ticker to `set_xticker`.
        If *False*, let :obj:`~matplotlib.figure.Figure` instance decide.
    set_yticker : int, float or False. Default: False
        If int or float, sets the y-axis ticker to `set_yticker`.
        If *False*, let :obj:`~matplotlib.figure.Figure` instance decide.
    ax : :obj:`~matplotlib.axes._axes.Axes` object or None. Default: None
        If :obj:`~matplotlib.axes._axes.Axes` object, centers the axis spines
        of specified :obj:`~matplotlib.figure.Figure` instance.
        If *None*, centers the axis spines of current
        :obj:`~matplotlib.figure.Figure` instance.

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
        # TODO: STILL HAVE TO FIX THAT THE TICKLABELS ARE ALWAYS HIDDEN
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
    if not set_xticker:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(set_xticker))

    # Set y-axis ticker
    if not set_yticker:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(set_yticker))


class CenteredFormatter(mpl.ticker.ScalarFormatter):
    """
    Acts exactly like the default Scalar Formatter, but yields an empty
    label for ticks at "center".

    """

    center = 0

    def __call__(self, value, pos=None):
        if(value == self.center):
            return("")
        else:
            return(mpl.ticker.ScalarFormatter.__call__(self, value, pos))


def draw_textline(text, x=None, y=None, pos='start top', ax=None,
                  line_kwargs={}, text_kwargs={}):
    """
    Draws a line on the axis `ax` in a :obj:`~matplotlib.figure.Figure`
    instance and prints `text` on top.

    Parameters
    ----------
    text : str
        Text to be printed on the line.
    x : scalar or None
        If scalar, text/line x-coordinate.
        If *None*, line covers complete x-axis.
        Either `x` or `y` needs to be *None*.
    y : scalar or None
        If scalar, text/line y-coordinate.
        If *None*, line covers complete y-axis.
        Either `x` or `y` needs to be *None*.

    Optional
    --------
    pos : {'start', 'end'}{'top', 'bottom'}. Default: 'start top'
        If 'start', prints the text at the start of the drawn line.
        If 'end', prints the text at the end of the drawn line.
        If 'top', prints the text above the drawn line.
        If 'bottom', prints the text below the drawn line.
        Arguments must be given as a single string.
    ax : :obj:`~matplotlib.axes._axes.Axes` object or None. Default: None
        If :obj:`~matplotlib.axes._axes.Axes` object, draws line in specified
        :obj:`~matplotlib.figure.Figure` instance.
        If *None*, draws line in current :obj:`~matplotlib.figure.Figure`
        instance.
    line_kwargs : dict of :func:`~matplotlib.lines.Line2D` properties.\
        Default: {}
        The keyword arguments used for drawing the line.
    text_kwargs : dict of :func:`~matplotlib.text.Text` properties. Default: {}
        The keyword arguments used for drawing the text.

    """

    # If no AxesSubplot object is provided, make one
    if ax is None:
        ax = plt.gca()

    # Set default line_kwargs and text_kwargs
    default_line_kwargs = {'linestyle': '-',
                           'color': 'k'}
    default_text_kwargs = {'color': 'k',
                           'fontsize': 14}

    # Combine given kwargs with default ones
    default_line_kwargs.update(line_kwargs)
    default_text_kwargs.update(text_kwargs)
    line_kwargs = default_line_kwargs
    text_kwargs = default_text_kwargs

    # Check if certain keyword arguments are present in text_fmt
    for key, val in text_kwargs.items():
        if key in ('va', 'ha', 'verticalalignment', 'horizontalalignment',
                   'rotation'):
            text_kwargs.pop(key)

    if x is None and y is not None:
        # Adjust axes to include text in plot
        ax_ysize = abs(ax.set_ylim()[1]-ax.set_ylim()[0])

        # Adjust axes if line is located on the bottom
        if(ax.set_ylim()[0] > y):
            ax.set_ylim(y, ax.set_ylim()[1])
        if(ax.set_ylim()[0] <= y and ax.set_ylim()[0] >= y-0.1*ax_ysize):
            ax.set_ylim(y-0.1*ax_ysize, ax.set_ylim()[1])
        elif(ax.set_ylim()[0] <= y and ax.set_ylim()[0] >= y+0.1*ax_ysize):
            ax.set_ylim(y+0.1*ax_ysize, ax.set_ylim()[1])

        # Adjust axes if line is located on the top
        if(ax.set_ylim()[1] < y):
            ax.set_ylim(ax.set_ylim()[0], y)
        if(ax.set_ylim()[1] >= y and ax.set_ylim()[1] <= y+0.1*ax_ysize):
            ax.set_ylim(ax.set_ylim()[0], y+0.1*ax_ysize)
        elif(ax.set_ylim()[1] >= y and ax.set_ylim()[1] <= y-0.1*ax_ysize):
            ax.set_ylim(ax.set_ylim()[0], y-0.1*ax_ysize)

        # Draw line
        ax.plot(ax.set_xlim(), [y, y], **line_kwargs)

        # Gather axis specific text properties
        x = ax.set_xlim()[0]
        y = y
        rotation = 0

        # Gather case specific text properties
        if ('start') in pos.lower() and ('top') in pos.lower():
            ha = 'left'
            va = 'bottom'
        elif ('start') in pos.lower() and ('bottom') in pos.lower():
            ha = 'left'
            va = 'top'
        elif ('end') in pos.lower() and ('top') in pos.lower():
            ha = 'right'
            va = 'bottom'
        elif ('end') in pos.lower() and ('bottom') in pos.lower():
            ha = 'right'
            va = 'top'
        else:
            raise ValueError("Input argument 'pos' is invalid!")

    elif y is None and x is not None:
        # Adjust axes to include text in plot
        ax_xsize = abs(ax.set_xlim()[1]-ax.set_xlim()[0])

        # Adjust axes if line is located on the left
        if(ax.set_xlim()[0] > x):
            ax.set_xlim(x, ax.set_xlim()[1])
        if(ax.set_xlim()[0] <= x and ax.set_xlim()[0] >= x-0.1*ax_xsize):
            ax.set_xlim(x-0.1*ax_xsize, ax.set_xlim()[1])
        elif(ax.set_xlim()[0] <= x and ax.set_xlim()[0] >= x+0.1*ax_xsize):
            ax.set_xlim(x+0.1*ax_xsize, ax.set_xlim()[1])

        # Adjust axes if line is located on the right
        if(ax.set_xlim()[1] < x):
            ax.set_xlim(ax.set_xlim()[0], x)
        if(ax.set_xlim()[1] >= x and ax.set_xlim()[1] <= x+0.1*ax_xsize):
            ax.set_xlim(ax.set_xlim()[0], x+0.1*ax_xsize)
        elif(ax.set_xlim()[1] >= x and ax.set_xlim()[1] <= x-0.1*ax_xsize):
            ax.set_xlim(ax.set_xlim()[0], x-0.1*ax_xsize)

        # Draw line
        ax.plot([x, x], ax.set_ylim(), **line_kwargs)

        # Gather axis specific text properties
        x = x
        y = ax.set_ylim()[0]
        rotation = 90

        # Gather case specific text properties
        if ('start') in pos.lower() and ('top') in pos.lower():
            ha = 'right'
            va = 'bottom'
        elif ('start') in pos.lower() and ('bottom') in pos.lower():
            ha = 'left'
            va = 'bottom'
        elif ('end') in pos.lower() and ('top') in pos.lower():
            ha = 'right'
            va = 'top'
        elif ('end') in pos.lower() and ('bottom') in pos.lower():
            ha = 'left'
            va = 'top'
        else:
            raise ValueError("Input argument 'pos' is invalid!")

    else:
        raise InputError("Either of input arguments 'x' and 'y' needs to be "
                         "*None*!")

    # Draw text
    ax.text(x, y, text, rotation=rotation, ha=ha, va=va, **text_kwargs)


def f2tex(value, sdigits=4, power=3, nobase1=True):
    """
    Transform a value into a (La)TeX string for usage in a
    :obj:`~matplotlib.figure.Figure` instance.

    Parameters
    ----------
    value : int or float
        Value to be transformed.

    Optional
    --------
    sdigits : int. Default: 4
        Maximum amount of significant digits `value` is returned with.
    power : int. Default: 3
        Minimum log10(`value`) required before `value` is written in
        scientific form.
    nobase1 : bool. Default: True
        Whether or not to include `base` in scientific form if `base=1`.

    Returns
    -------
    out : string
        String containing `value` written in (La)TeX string.

    Examples
    --------
    >>> f2tex(20.2935826592)
    '20.29'


    >>> f2tex(20.2935826592, sdigits=6)
    '20.2936'


    >>> f2tex(20.2935826592, power=1)
    '2.029\\\\cdot 10^{1}'


    >>> f2tex(1e6, nobase1=True)
    '10^{6}'


    >>> f2tex(1e6, nobase1=False)
    '1\\\\cdot 10^{6}'

    """

    # If value is zero, it cannot be converted to a log
    if(value == 0):
        return('0')
    else:
        n = int(np.floor(np.log10(abs(value))))

    if(abs(n) < power):
        string = r"{0:.{1}g}".format(value, sdigits)
    else:
        base = value/pow(10, n)
        if(base == 1 and nobase1):
            string = r"10^{{{0}}}".format(n)
        else:
            string = r"{0:.{1}g}\cdot 10^{{{2}}}".format(base, sdigits, n)
    return(string)


def q2tex(quantity, sdigits=4, power=3, nobase1=True, unitfrac=False):
    """
    Combination of :func:`~e13tools.pyplot.f2tex` and
    :func:`~e13tools.pyplot.apu2tex`.

    Transform a :obj:`~astropy.units.quantity.Quantity` object into a (La)TeX
    string for usage in a :obj:`~matplotlib.figure.Figure` instance.

    Parameters
    ----------
    quantity : int, float or :obj:`~astropy.units.quantity.Quantity` object
        Quantity to be transformed.

    Optional
    --------
    sdigits : int. Default: 4
        Maximum amount of significant digits `value` is returned with.
    power : int. Default: 3
        Minimum log10(`value`) required before `value` is written in
        scientific form.
    nobase1 : bool. Default: True
        Whether or not to include `base` in scientific form if `base=1`.
    unitfrac : bool. Default: False
        Whether or not to write `unit` as a LaTeX fraction.

    Returns
    -------
    out : string
        String containing `quantity` written in (La)TeX string.

    Examples
    --------
    >>> import astropy.units as apu
    >>> q2tex(20.2935826592)
    '20.29'


    >>> import astropy.units as apu
    >>> q2tex(20.2935826592*apu.solMass/apu.yr)
    '20.29\\\\ \\\\mathrm{M_{\\\\odot}\\\\,yr^{-1}}'

    >>> import astropy.units as apu
    >>> q2tex(20.2935826592*apu.solMass/apu.yr, sdigits=6)
    '20.2936\\\\ \\\\mathrm{M_{\\\\odot}\\\\,yr^{-1}}'


    >>> import astropy.units as apu
    >>> q2tex(20.2935826592*apu.solMass/apu.yr, power=1)
    '2.029\\\\cdot 10^{1}\\\\ \\\\mathrm{M_{\\\\odot}\\\\,yr^{-1}}'


    >>> import astropy.units as apu
    >>> q2tex(1e6*apu.solMass/apu.yr, nobase1=True)
    '10^{6}\\\\ \\\\mathrm{M_{\\\\odot}\\\\,yr^{-1}}'


    >>> import astropy.units as apu
    >>> q2tex(1e6*apu.solMass/apu.yr, nobase1=False)
    '1\\\\cdot 10^{6}\\\\ \\\\mathrm{M_{\\\\odot}\\\\,yr^{-1}}'


    >>> import astropy.units as apu
    >>> q2tex(20.2935826592*apu.solMass/apu.yr, unitfrac=False)
    '20.29\\\\ \\\\mathrm{M_{\\\\odot}\\\\,yr^{-1}}'


    >>> import astropy.units as apu
    >>> q2tex(20.2935826592*apu.solMass/apu.yr, unitfrac=True)
    '20.29\\\\ \\\\mathrm{\\\\frac{M_{\\\\odot}}{yr}}'

    """

    # Check if quantity has a unit
    if import_astropy:
        if isinstance(quantity, apu.quantity.Quantity):
            value = quantity.value
            unit = quantity.unit
        else:
            value = quantity
            unit = 0

        # Value handling
        string = f2tex(value, sdigits, power, nobase1)

        # Unit handling
        if unit:
            unit_string = apu2tex(unit, unitfrac)
            string = ''.join([string, '\ ', unit_string])

        return(string)
    else:
        f2tex(quantity, sdigits, power, nobase1)


def suplabel(label, axis, pos='min', labelpad=9, fig=None, **kwargs):
    """
    Adds a super label in the provided figure `fig` for the specified `axis`.
    Works similarly to :meth:`~matplotlib.pyplot.Figure.suptitle`, but for axes
    labels instead of figure titles.

    This algorithm is based on a Stack Overflow answer by KYC [1]_.

    Parameters
    ----------
    label : str
        The text to be used as the axis label.
    axis : {'x', 'y'}
        String indicating which axis will receive the created label.

    Optional
    --------
    pos : {'min', 'max'}. Default: 'min'
        String indicating whether to position the axis label at the minimum or
        maximum of the opposing axis. If 'min', the axis label will be
        positioned on the left (if `axis` = 'y') or below (if `axis` = 'x') the
        figure. If 'max', the axis label will be positioned on the right (if
        `axis` = 'y') or above (if `axis` = 'x') the figure.
    labelpad : float. Default: 9
        Distance/padding between the `axis` and the label.
    fig : :obj:`~matplotlib.figure.Figure` object or None. Default: None
        In which :obj:`~matplotlib.figure.Figure` object the axis label needs
        to be drawn. If *None*, the current :obj:`~matplotlib.figure.Figure`
        object will be used.
    kwargs : dict of :func:`~matplotlib.text.Text` properties. Default: {}
        The keyword arguments used for drawing the text.

    References
    ----------
    .. [1] https://stackoverflow.com/a/29107972

    """

    # Obtain a reference to the current figure if not provided
    if fig is None:
        fig = plt.gcf()

    # Create empty lists of x and y extrema
    xmin = []
    xmax = []
    ymin = []
    ymax = []

    # Obtain x and y extrema
    for ax in fig.axes:
        xmin.append(ax.get_position().xmin)
        xmax.append(ax.get_position().xmax)
        ymin.append(ax.get_position().ymin)
        ymax.append(ax.get_position().ymax)

    # Get positions of all corners of the figure
    xmin = min(xmin)
    xmax = max(xmax)
    ymin = min(ymin)
    ymax = max(ymax)

    # Check if any value for horizontalalignment or verticalalignment is given
    try:
        kwargs['ha']
    except KeyError:
        try:
            kwargs['horizontalalignment']
        except KeyError:
            kwargs['ha'] = 'center'

    try:
        kwargs['va']
    except KeyError:
        try:
            kwargs['verticalalignment']
        except KeyError:
            kwargs['va'] = 'center'

    # Get all properties for axis label
    if(axis.lower() == 'x'):
        x = 0.5
        rotation = 0
        if(pos.lower() == 'min'):
            y = ymin-float(labelpad)/fig.dpi
        elif(pos.lower() == 'max'):
            y = ymax+float(labelpad)/fig.dpi
        else:
            raise InputError("Input argument 'pos' is invalid!")
    elif(axis.lower() == 'y'):
        y = 0.5
        rotation = 90
        if(pos.lower() == 'min'):
            x = xmin-float(labelpad)/fig.dpi
        elif(pos.lower() == 'max'):
            x = xmax+float(labelpad)/fig.dpi
        else:
            raise InputError("Input argument 'pos' is invalid!")
    else:
        raise InputError("Input argument 'axis' is invalid!")

    # Draw axis label
    fig.text(x, y, label, rotation=rotation, **kwargs)
