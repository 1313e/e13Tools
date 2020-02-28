# -*- coding: utf-8 -*-

"""
PyPlot
======
Provides a collection of functions useful in various plotting routines.

"""


# %% IMPORTS
# Package imports
try:
    import astropy.units as apu
    import_astropy = 1
except ImportError:  # pragma: no cover
    import_astropy = 0
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# e13Tools imports
from e13tools.core import InputError

# All declaration
__all__ = ['apu2tex', 'center_spines', 'draw_textline', 'f2tex', 'q2tex']


# %% FUNCTIONS
# This function converts an astropy unit into a TeX string
def apu2tex(unit, unitfrac=False):
    """
    Transform a :obj:`~astropy.units.Unit` object into a (La)TeX string for
    usage in a :obj:`~matplotlib.figure.Figure` instance.

    Parameters
    ----------
    unit : :obj:`~astropy.units.Unit` object
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
    '\\mathrm{M_{\\odot}}'

    >>> import astropy.units as apu
    >>> apu2tex(apu.solMass/apu.yr, unitfrac=False)
    '\\mathrm{M_{\\odot}\\,yr^{-1}}'

    >>> import astropy.units as apu
    >>> apu2tex(apu.solMass/apu.yr, unitfrac=True)
    '\\mathrm{\\frac{M_{\\odot}}{yr}}'

    """

    if import_astropy:
        if not unitfrac:
            string = unit.to_string('latex_inline')
        else:
            string = unit.to_string('latex')

        # Remove '$' from the string and make output a string (py2.7)
        return(str(string.replace("$", "")))
    else:  # pragma: no cover
        raise ImportError("This function requires AstroPy!")


# This function centers the axes of the provided axes
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
    ax : :obj:`~matplotlib.axes.Axes` object or None. Default: None
        If :obj:`~matplotlib.axes.Axes` object, centers the axis spines
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
        formatter = mpl.ticker.ScalarFormatter()
        formatter.center = center
        axis.set_major_formatter(formatter)

    # Add origin offset ticklabel if <centerx=0, centery=0> using annotation
    if(centerx == 0 and centery == 0):
        xlabel, ylabel = map(formatter.format_data, [centerx, centery])
        ax.annotate("0", (centerx, centery), xytext=(-4, -4),
                    textcoords='offset points', ha='right', va='top')

    # Set x-axis ticker
    if set_xticker:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(set_xticker))

    # Set y-axis ticker
    if set_yticker:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(set_yticker))


# This function draws a line with text in the provided figure
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
    ax : :obj:`~matplotlib.axes.Axes` object or None. Default: None
        If :obj:`~matplotlib.axes.Axes` object, draws line in specified
        :obj:`~matplotlib.figure.Figure` instance.
        If *None*, draws line in current :obj:`~matplotlib.figure.Figure`
        instance.
    line_kwargs : dict of :class:`~matplotlib.lines.Line2D` properties.\
        Default: {}
        The keyword arguments used for drawing the line.
    text_kwargs : dict of :class:`~matplotlib.text.Text` properties.\
        Default: {}
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
    text_keys = list(text_kwargs.keys())
    for key in text_keys:
        if key in ('va', 'ha', 'verticalalignment', 'horizontalalignment',
                   'rotation'):
            text_kwargs.pop(key)

    # Set line specific variables
    if x is None and y is not None:
        ax_set_lim = ax.set_ylim
        other_ax_lim = ax.set_xlim
        axis = y
        draw_axes = (other_ax_lim(), [y, y])

    elif x is not None and y is None:
        ax_set_lim = ax.set_xlim
        other_ax_lim = ax.set_ylim
        axis = x
        draw_axes = ([x, x], other_ax_lim())

    else:
        raise InputError("Either of input arguments 'x' and 'y' needs to be "
                         "*None*!")

    # Obtain length of selected axis
    ax_size = abs(ax_set_lim()[1]-ax_set_lim()[0])

    # Adjust axes if line is located on the bottom/left
    if(ax_set_lim()[0] > axis):
        ax_set_lim(axis, ax_set_lim()[1])
    if(ax_set_lim()[0] <= axis and ax_set_lim()[0] >= axis-0.1*ax_size):
        ax_set_lim(axis-0.1*ax_size, ax_set_lim()[1])

    # Adjust axes if line is located on the top/right
    if(ax_set_lim()[1] < axis):
        ax_set_lim(ax_set_lim()[0], axis)
    if(ax_set_lim()[1] >= axis and ax_set_lim()[1] <= axis+0.1*ax_size):
        ax_set_lim(ax_set_lim()[0], axis+0.1*ax_size)

    # Draw line
    ax.plot(*draw_axes, **line_kwargs)

    # Gather case specific text properties
    if ('start') in pos.lower() and ('top') in pos.lower():
        ha = 'left' if x is None else 'right'
        va = 'bottom'
        other_axis = other_ax_lim()[0]
    elif ('start') in pos.lower() and ('bottom') in pos.lower():
        ha = 'left'
        va = 'top' if x is None else 'bottom'
        other_axis = other_ax_lim()[0]
    elif ('end') in pos.lower() and ('top') in pos.lower():
        ha = 'right'
        va = 'bottom' if x is None else 'top'
        other_axis = other_ax_lim()[1]
    elif ('end') in pos.lower() and ('bottom') in pos.lower():
        ha = 'right' if x is None else 'left'
        va = 'top'
        other_axis = other_ax_lim()[1]
    else:
        raise ValueError("Input argument 'pos' is invalid!")

    # Set proper axes and rotation
    if x is None:
        x = other_axis
        rotation = 0
    else:
        y = other_axis
        rotation = 90

    # Draw text
    ax.text(x, y, text, rotation=rotation, ha=ha, va=va, **text_kwargs)


# This function converts a float into a TeX string
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
    '2.029\\cdot 10^{1}'


    >>> f2tex(1e6, nobase1=True)
    '10^{6}'


    >>> f2tex(1e6, nobase1=False)
    '1\\cdot 10^{6}'

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


# This function converts an astropy quantity into a TeX string
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
    '20.29\\,\\mathrm{M_{\\odot}\\,yr^{-1}}'


    >>> import astropy.units as apu
    >>> q2tex(20.2935826592*apu.solMass/apu.yr, sdigits=6)
    '20.2936\\,\\mathrm{M_{\\odot}\\,yr^{-1}}'


    >>> import astropy.units as apu
    >>> q2tex(20.2935826592*apu.solMass/apu.yr, power=1)
    '2.029\\cdot 10^{1}\\,\\mathrm{M_{\\odot}\\,yr^{-1}}'


    >>> import astropy.units as apu
    >>> q2tex(1e6*apu.solMass/apu.yr, nobase1=True)
    '10^{6}\\,\\mathrm{M_{\\odot}\\,yr^{-1}}'


    >>> import astropy.units as apu
    >>> q2tex(1e6*apu.solMass/apu.yr, nobase1=False)
    '1\\cdot 10^{6}\\,\\mathrm{M_{\\odot}\\,yr^{-1}}'


    >>> import astropy.units as apu
    >>> q2tex(20.2935826592*apu.solMass/apu.yr, unitfrac=False)
    '20.29\\,\\mathrm{M_{\\odot}\\,yr^{-1}}'


    >>> import astropy.units as apu
    >>> q2tex(20.2935826592*apu.solMass/apu.yr, unitfrac=True)
    '20.29\\,\\mathrm{\\frac{M_{\\odot}}{yr}}'

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
            string = ''.join([string, r'\,', unit_string])

        return(string)
    else:  # pragma: no cover
        f2tex(quantity, sdigits, power, nobase1)
